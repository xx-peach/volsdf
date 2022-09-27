import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler

class ImplicitNetwork(nn.Module):
    """ This is a Network to Predict Sample Point's SDF """
    def __init__(
            self,
            feature_vector_size,    # feature vector dimension of sdfnet, default = 256
            sdf_bounding_sphere,    #?还不知道这是干啥的, default = 3.0
            d_in,                   # input channel, default = 3
            d_out,                  # output channel, default = 1
            dims,                   # channels for hidden layers, default = [256, 256, 256, ...]
            geometric_init=True,
            bias=1.0,
            skip_in=(),             # similar to NeRF, resblock index, default = [4]
            weight_norm=True,
            multires=0,             # maximum exponential of position encoding, default = 6
            sphere_scale=1.0,       #?还不知道这是干啥的, default = 20.0
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale

        # each layer's channel from input to output
        dims = [d_in] + dims + [d_out + feature_vector_size]

        # instantiate the embedder for input sample points
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)     # total number of layer in sdfnet
        self.skip_in = skip_in          # every skip_in layers is a reslayer

        # create the ModuleList for sdfnet
        for l in range(0, self.num_layers - 1):
            # reduce the current output channel if next layer is the reslayer
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            # instantiate the middle layer using dims[l] as input channel and out_dim as output channel
            lin = nn.Linear(dims[l], out_dim)
            # initialize the weight and bias of each layer if 'geometric_init' == True 
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            # normalize the initial weights of current layer if 'weight_norm' == True
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            # set lin as attribute 'lin{}' of ImplicitNetwork
            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        # add positional encoding to raw input sample points
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        
        # go through the sdfnet
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def get_outputs(self, x):
        """ External Interface, Called by VolSDFNetwork()
        Arguments:
            x - (batch_size*H*W*(N_samples+2+N_samples_extra), 3), input sample points, in world coordinate
        Returns:
            sdf             - (batch_size*H*W*(N_samples+2+N_samples_extra),   1), sdf of each input sample point
            feature_vectors - (batch_size*H*W*(N_samples+2+N_samples_extra), 256), feature vectors for rendering network
            gradients       - #?(batch_size*H*W*(N_samples+2+N_samples_extra),   1), each sample point's normal direction, used in NeRF
        """
        x.requires_grad_(True)
        output = self.forward(x)            # (batch_size*H*W*(N_samples+2+N_samples_extra), 257), sdf + feature_vectors
        # fetch sdf from implicit network output
        sdf = output[:, :1]                 # (batch_size*H*W*(N_samples+2+N_samples_extra), 1), sdf
        # clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        # fetch feature_vectors from implicit network output
        feature_vectors = output[:, 1:]     # (batch_size*H*W*(N_samples+2+N_samples_extra), 256), feature_vectors
        # 求输出 sdf 对输入 x 的导数 gradients(这个输入是做完 algorithm 1 之后选的 sample points)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]            # (batch_size*H*W*(N_samples+2+N_samples_extra), 1)

        return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        """ Function Called by ErrorBoundSampler() within a torch.no_grad() Segment, Algorithm 1 采样时用的函数
        Arguments:
            x - (batch_size*H*W*N_samples, 3), sampled points of current iteration, in world coordinate
        Returns:
            sdf - (batch_size*H*W*N_samples, 1), sdf of each input sample point
        """
        sdf = self.forward(x)[:, :1]    # (batch_size*H*W*N_samples, 1)
        # clamping the sdf with the scene bounding sphere, so that all rays are eventually occluded
        if self.sdf_bounding_sphere > 0.0:
            #? self.sphere_scale = 20.0, self.sdf_bounding_sphere = 3.0, 这个 bounding sphere 真的要这么大吗
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2, 1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        """ forward() function of rendering network
        Arguments:
            points          - (batch_size*H*W*(N_samples+2+N_samples_extra), 3), better sample points from algorithm 1
            normals         - (batch_size*H*W*(N_samples+2+N_samples_extra), 3), normal direction of each sample point
            view_dirs       - (batch_size*H*W*(N_samples+2+N_samples_extra), 3), view directions of each sample point
            feature_vectors - (batch_size*H*W*(N_samples+2+N_samples_extra), 256), feature vectors from implicit network
        Returns:
            x - (batch_size*H*W*(N_samples+2+N_samples_extra), 3), output rgb color of each sample point for volume rendering
        """
        if self.embedview_fn is not None:
            #TODO points 在这不用做 positional encoding, 但是在 implicit network 里面做了
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            #! VolSDF 中说到的区别与传统的 NeRF, 这边还把每个点的 normal direction 考虑进来(BRDF's model)
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x

class VolSDFNetwork(nn.Module):
    """VolSDF Network including Implicit Network and RenderingNetwork"""
    def __init__(self, conf):
        super().__init__()
        # get some basic configurations
        self.feature_vector_size = conf.get_int('feature_vector_size')                      # default = 256
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)   # default = 3.0
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)                        # default = False
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda() # default background color = [1, 1, 1]
        # instantiate two basic networks, namely implicit network for sdf and rendering network for rgb
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        # instantiate density model sigma = alpha * Laplace(-sdf)
        self.density = LaplaceDensity(**conf.get_config('density'))
        # instantiate a sophisticated sampler
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def forward(self, input):
        """ VolSDF Network's forwad() Function
        Arguments:
            model_input["intrinsics"] - (batch_size, 4, 4)
            model_input["uv"]         - (batch_size, H*W, 2)
            model_input['pose']       - (batch_size, 4, 4)
        Returns:
            output - (bs, )
        """
        # parse model input: camera intrinsic, extrinsic and grid coordinate of current image
        intrinsics = input["intrinsics"]    # (batch_size, 4, 4)
        uv = input["uv"]                    # (batch_size, H*W, 2)
        pose = input["pose"]                # (batch_size, 4, 4)

        # get rays direction and origin of all pixels, returns are in the world coordinate
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)   # (batch_size, H*W, 3), (batch_size, 3)
        batch_size, num_pixels, _ = ray_dirs.shape

        # duplicate rays origin and reshape both origin and direction to shape (batch_size*H*W, 3)
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)          # (batch_size*H*W, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)                                              # (batch_size*H*W, 3)

        #! sophisticated sampling algorithm proposed in paper，迭代优化采样点来近似 opacity approximation
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)    # (batch_size*H*W, N_samples+2+N_samples_extra)
        N_samples = z_vals.shape[1]                                 # N_samples+2+N_samples_extra
        # prepare world coordinates of all the final sample points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)     # (batch_size*H*W, N_samples+2+N_samples_extra, 3)
        points_flat = points.reshape(-1, 3)                         # (batch_size*H*W*(N_samples+2+N_samples_extra), 3)
        # prepare(duplicate) view direcions for all the final sample points
        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)        # (batch_size*H*W, N_samples+2+N_samples_extra, 3)
        dirs_flat = dirs.reshape(-1, 3)                             # (batch_size*H*W*(N_samples+2+N_samples_extra), 3)
        # compute each sample point's sdf using implicit network
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)# (batch_size*H*W*(N_samples+2+N_samples_extra), 1/256)

        # a standard NeRF architecture, output egb: (batch_size*H*W*(N_samples+2+N_samples_extra), 3)
        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)                    # (batch_size*H*W, N_samples+2+N_samples_extra, 3)

        # compute weight of each sample point
        weights = self.volume_rendering(z_vals, sdf)                # (batch_size*H*W, N_samples+2+N_samples_extra)
        # render rgb color using weight
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)      # (batch_size*H*W, 3)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb_values': rgb_values,
        }

        if self.training:
            # sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels      # batch_size * H * W
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()
            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta

        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        """ Compute `Weight` for each Sample Point
        Arguments:
            z_vals - (batch_size*H*W, N_samples+2+N_samples_extra), z-axis value of each sample point
            sdf    - (batch_size*H*W*(N_samples+2+N_samples_extra), 1), each sample point's sdf
        Returns:
            weights - (batch_size*H*W, N_samples+2+N_samples_extra), probability of the ray hits something here
        """
        density_flat = self.density(sdf)                        # (batch_size*H*W, N_samples+2+N_samples_extra, 1)
        density = density_flat.reshape(-1, z_vals.shape[1])     # (batch_size*H*W, N_samples+2+N_samples_extra)

        dists = z_vals[:, 1:] - z_vals[:, :-1]                  # (batch_size*H*W, N_samples+2+N_samples_extra-1)
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density                           # (batch_size*H*W, N_samples+2+N_samples_extra)
        # shift one step, shape of (batch_size*H*W, N_samples+2+N_samples_extra)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
        # probability of it is not empty here
        alpha = 1 - torch.exp(-free_energy)                     # (batch_size*H*W, N_samples+2+N_samples_extra)
        # probability of everything is empty up to now
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        # probability of the ray hits something here
        weights = alpha * transmittance

        return weights
