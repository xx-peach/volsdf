import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 ):
        # get the truely data directory path from input configuration
        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        # get the image's resolution
        self.total_pixels = img_res[0] * img_res[1]         # total_pixels = H * W
        self.img_res = img_res                              # img_res = (H, W)

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        # get a list of sorted image names and total number of images
        image_dir = '{0}/image'.format(self.instance_dir)   # image path, eg. '../data/DTU/scan65/image'
        image_paths = sorted(utils.glob_imgs(image_dir))    # sorted list of all image names
        self.n_images = len(image_paths)                    # total number of all images

        # get camera parameters(还不知道它给的 npz 里面存的是什么相机参数) for each image
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # decomposite camera's intrinsic and extrinsic matrix from raw parameter for each image
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)              # (4, 4), (4, 4)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())    # [(4, 4), ..., (4, 4)]
            self.pose_all.append(torch.from_numpy(pose).float())                # [(4, 4), ..., (4, 4)]

        # read all the images in
        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)              # (H, W, 3)
            rgb = rgb.reshape(3, -1).transpose(1, 0)    # (H* W, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())               # [(H*W, 3), ..., (H*W, 3)]

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """ Custom __getitem__() Function
        Arguments:
            idx - int, input index from sampler()
        Returns:
            idx               - int, input index from sampler()
            sample.uv         - (H*W, 2), 2d grid image indices, `eg.` [[0, 0], [1, 0], [2, 0], ...]
            sample.intrinsics - (4, 4), camera intrinsic of idx-th image
            sample.pose       - (4, 4), camera extrinsic of idx-th image
            ground_truth.rgb  - (H*W, 3), original idx-th image
        """
        # generate 2d grid coordinates
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,                               # (H*W, 2), [[0, 0], [1, 0], [2, 0], ...]
            "intrinsics": self.intrinsics_all[idx], # (4, 4), camera intrinsic
            "pose": self.pose_all[idx]              # (4, 4), camera extrinsic
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]             # (H*W, 3)
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
