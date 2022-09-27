import abc
import torch

from utils import rend_util

class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self,near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass

class UniformSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1):
        super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    def get_z_vals(self, ray_dirs, cam_loc, model):
        """ Uniform Sampling Strategy as Initialize
        Arguments:
            ray_dirs - (batch_size*H*W, 3), rays direction in world coordinate
            cam_loc  - (batch_size*H*W, 3), rays origin in world coordinate
            model    - VolSDFNetwork() model, we nned its ImplicitNetwork() to predict sdf
        Returns:
            z_values      - (N_samples,), z-axis value of each sample point, uniform
                            sample 就是从 N_samples 个均匀区间中随机采样, 并不是刚好是格点
        """
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            sphere_intersections = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            far = sphere_intersections[:,1:]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()    # (N_samples,)
        z_vals = near * (1. - t_vals) + far * (t_vals)                  # (batch_size*H*W, N_samples)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])    # (batch_size*H*W, N_samples-1)
            upper = torch.cat([mids, z_vals[..., -1:]], -1)     # (batch_size*H*W, N_samples)
            lower = torch.cat([z_vals[..., :1], mids], -1)      # (batch_size*H*W, N_samples)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()            # (batch_size*H*W, N_samples)
            z_vals = lower + (upper - lower) * t_rand           # (batch_size*H*W, N_samples)

        return z_vals


class ErrorBoundSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval, take_sphere_intersection=inverse_sphere_bg)

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, model):
        """ sophisticated sampling algorithm proposed in paper
        Arguments:
            ray_dirs - (batch_size*H*W, 3), rays direction in world coordinate
            cam_loc  - (batch_size*H*W, 3), rays origin in world coordinate
            model    - VolSDFNetwork() model, we nned its ImplicitNetwork() to predict sdf
        Returns:
            z_values      - (batch_size*H*W, N_samples+2+N_samples_extra), final sample points, in world coordinate
            z_samples_eik - (batch_size*H*W, batch_size*H*W), #? some points near surface, 我不理解
        """
        # get initial beta, default = 0.1
        beta0 = model.density.get_beta().detach()
        ###############################
        # start with uniform sampling #
        ###############################
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, model)  # (batch_size*H*W, N_samples)
        samples, samples_idx = z_vals, None
        ###################################################
        # get maximum beta from the upper bound (Lemma 2) #
        ###################################################
        dists = z_vals[:, 1:] - z_vals[:, :-1]                              # (batch_size*H*W, N_samples-1)
        #! note: (1) 这里假设 alpha = 1/beta, 所以后面要开方, (2) M^2/(n-1) == n-1 个 (M/n)^2 相加
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)                                            # (batch_size*H*W,), 这是 initialized 的 beta_+
        #################################
        # implementation of algorithm 1 #
        #################################
        total_iters, not_converge = 0, True
        while not_converge and total_iters < self.max_total_iters:
            # sample N_sample points from each ray
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)    # (batch_size*H*W, N_samples, 3)
            points_flat = points.reshape(-1, 3)                                             # (batch_size*H*W*N_samples, 3)
            #######################################################
            # calculating the SDF only for the new sampled points #
            #######################################################
            #! 这边只是计算出 sample points 的 sdf 用来做 opacity approximation，不是在这里训 ImplicitNetwork 的
            with torch.no_grad():
                samples_sdf = model.implicit_network.get_sdf_vals(points_flat)              # (batch_size*H*W*N_samples, 1)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf       # (batch_size*H*W*N_samples, 1)
            #################################################################
            # calculating the bound d* (Theorem 1), 算 d* 是后面来算 E(t) 用的 #
            #################################################################
            d = sdf.reshape(z_vals.shape)                       # (batch_size*H*W, N_samples)
            dists = z_vals[:, 1:] - z_vals[:, :-1]              # (batch_size*H*W, N_samples-1)
            # get delta_i, |d_i|, |d_{i+1}| respectively, namely a, b, c
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()    # a: delta_i = t_{i+1} - t_i, b: |d_i|, c: |d_{i+1}|
            # first, second 都是 case 2: |b^2 - c^2| >= a^2 的子情况，只不过是把绝对值拆开了而已
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)        # b^2 < c^2
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)       # b^2 > c^2
            # create the placeholder for d*, torch.zeros 是因为除了 case 1,2 剩下的情况 d* = 0
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()   # (batch_size*H*W, N_samples-1)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            # s_i = (delta_i + |d_i| + |d_{i+1}|) / 2
            s = (a + b + c) / 2.0
            # s_i * (s_i - a) * (s_i - b) * (s_i - c), 这是 h_i 的一部分
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            # mask 是 case 1: |d_i| + |d_{i+1}| <= delta_i
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            # fixing the sign
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star         # (batch_size*H*W, N_samples-1)
            #######################################################################
            # updating beta using line search, 对 batch_size * H * W 条 rays 同时算 #
            #######################################################################
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star) # (batch_size*H*W,)
            # 如果在 uniform sampling 的情况下 E(t) 就已经 < eps 了的话就直接把 beta 设置成 beta0(初始值)就行
            beta[curr_error <= self.eps] = beta0                                        # (batch_size*H*W,)
            # line search 去更新 beta, 这边采用的更新方法还挺高效的
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta       # (batch_size*H*W,)
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error >  self.eps] = beta_mid[curr_error >  self.eps]
            beta = beta_max
            #########################################################################################################
            # upsample more points, 用目前 sample points 的 sdf -> denisty -> weights -> cdf, 然后 inverse cdf 求新的点 #
            #########################################################################################################
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))     # (batch_size*H*W, N_samples)
            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)      # (batch_size*H*W, N_samples)
            free_energy = dists * density                                                   # (batch_size*H*W, N_samples)
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)   # (batch_size*H*W, N_samples+1)
            alpha = 1 - torch.exp(-free_energy)                                             # (batch_size*H*W, N_samples+1)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))           # (batch_size*H*W, N_samples+1)
            # probability of the ray hits something here
            weights = alpha * transmittance                                                 # (batch_size*H*W, N_samples+1)
            #######################################################
            #  check if we are done and this is the last sampling #
            #######################################################
            total_iters += 1                    # increment total_iters
            not_converge = beta.max() > beta0   # if one beta among batch_size * H * W is > beta0, 就是没收敛
            # sample more points proportional to the current error bound
            if not_converge and total_iters < self.max_total_iters:
                N = self.N_samples_eval
                bins = z_vals
                #! error_per_section 用的是 Equation 12, 只不过这里只对每个 [t_k, t_{k+1}] 区间进行计算, (batch_size*H*W, N_samples-1)
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:, :-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
                # add error_per_section together, 每个位置之前的累加和, (batch_size*H*W, N_samples-1)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                #! bound_opacity 用的是 Equation 13, 求得 error_integral 和 transmittance 之后 |O(t) - O'(t)| <= exp(-R'(t)) * (exp(E(t))-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[:, :-1]
                #* note 这边的 pdf 和 cdf 是关于 opacity approximation error, 是因为现在还没收敛/最大迭代, 要作为 upsampling points 的依据
                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            # sample the final sample set to be used in the volume rendering integral
            else:
                N = self.N_samples
                bins = z_vals
                #* note 这边的 pdf 和 cdf 是关于 volume rendering 公式中的 tau'(i) = (1-exp(delta_i * density_i)) * T(i) 的, 是最终 sample 的 O(t) 依据
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))
            ####################################################################
            # invert CDF sampling, 可能是上面两种情况中的一种, 具体加不加是后面再判断的 #
            ####################################################################
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()
            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
            #############################################################################
            # Adding samples if we not converged, add 的是 z_vals(不是直接 sample points) #
            #############################################################################
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        #############################
        # final sampled points here #
        #############################
        z_samples = samples                     # (batch_size*H*W, N_samples)
        near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda() # (batch_size*H*W, 1)
        #####################################################################################
        # if inverse sphere then need to add the far sphere intersection, #*default = False #
        #####################################################################################
        if self.inverse_sphere_bg:
            far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:, 1:]
        ###############################################################################
        # sample some extra points from algorithm 1 在迭代缩小 opacity error 时采的所有点 #
        ###############################################################################
        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)          # (batch_size*H*W, 2+N_samples_extra)
        # if self.N_samples_extra <= 0, 那就不用做这个 extra sampling from algorithm 1
        else:
            z_vals_extra = torch.cat([near, far], -1)                                   # (batch_size*H*W, 2)
        # 把 z_vals_extra 加到 z_samples 里面, 保证其中一定有 z_vals = near & far, 顺便把 extra 点也加进去(如果有的话)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)            # (batch_size*H*W, N_samples+2+N_samples_extra)/(batch_size*H*W, N_samples+2)
        ################################################################################
        # add some of the near surface points, #?不就是又多选了一些点吗，哪来的 near surface #
        ################################################################################
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()    # (batch_size*H*W,)
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))          # (batch_size*H*W, batch_size*H*W)
        #####################################################################################
        # if inverse sphere then need to add the far sphere intersection, #*default = False #
        #####################################################################################
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)   # (batch_size*H*W*N_samples, 1)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]


