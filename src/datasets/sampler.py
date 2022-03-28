import numpy as np
import torch

from torch_scatter import scatter_mean


class SampleManager():
    def __init__(self, img_h, img_w, patch_size=40):
        self.img_length = img_h * img_w
        self.img_h = img_h
        self.img_w = img_w
        self.caching_error_map = torch.zeros((img_h, img_w)) - 1.
        self.uniform_sample_ids = torch.randperm(self.img_length)
        self.num_uniform_samples = 0

        self.weighted_error_map = torch.ones((img_h, img_w)) * 5

        self.patch_size = patch_size
        self.coarse_img_h = int(img_h / patch_size)
        self.coarse_img_w = int(img_w / patch_size)
        self.weighted_error_map_coarse = torch.ones(
            (self.coarse_img_h, self.coarse_img_w)) * 5

    def reset(self):
        self.caching_error_map = torch.zeros((self.img_h, self.img_w)) - 1
        self.uniform_sample_ids = torch.randperm(self.img_length)
        self.num_uniform_samples = 0

    def log_error(self, uv, errors):
        """ store error in the cached error map

        Args:
            uv: [n, 2]
            errors: [n]
        """

        # make sure we are not overwriting
        assert np.all(self.caching_error_map[uv[:, 1], uv[:, 0]] == -1)
        self.caching_error_map[uv[:, 1], uv[:, 0]] = errors

    def log_weighted_error(self, uv, errors):
        self.weighted_error_map[uv[:, 1], uv[:, 0]] = errors

        coarse_uv = torch.floor(uv / self.patch_size).long()
        uv_1d = coarse_uv[:, 1] * self.coarse_img_w + coarse_uv[:, 0]
        unique_flat_ids, pinds, pcounts = torch.unique(
            uv_1d, return_inverse=True, return_counts=True)
        unique_y = unique_flat_ids // self.coarse_img_w
        unique_x = unique_flat_ids % self.coarse_img_w
        assert torch.max(unique_y) < self.coarse_img_h
        assert torch.max(unique_x) < self.coarse_img_w
        unique_uv = torch.stack([unique_x, unique_y], dim=-1)

        error_mean = scatter_mean(errors, pinds)
        self.weighted_error_map_coarse[unique_uv[:, 1], unique_uv[:, 0]] = error_mean

    def uniform_sample(self, num_samples):
        if self.num_uniform_samples + num_samples > self.img_length:
            print("reset sampler: {}".format(self.num_uniform_samples))
            self.reset()
        sampled_ids = self.uniform_sample_ids[
            self.num_uniform_samples: self.num_uniform_samples+num_samples]
        self.num_uniform_samples += num_samples
        return sampled_ids

    def weighted_sample(self, num_samples):
        """ sampling weighted by the error_maps
        """
        weights = self.weighted_error_map_coarse / torch.sum(self.weighted_error_map_coarse) + 1e-5

        coarse_uv_1d = torch.multinomial(
            weights.reshape(-1), num_samples=num_samples, replacement=True
        )  # [num_samples]
        coarse_y = coarse_uv_1d // self.coarse_img_w
        coarse_x = coarse_uv_1d % self.coarse_img_w
        rand_x = torch.randint(low=0, high=self.patch_size, size=(num_samples,))
        rand_y = torch.randint(low=0, high=self.patch_size, size=(num_samples,))

        x = coarse_x * self.patch_size + rand_x
        y = coarse_y * self.patch_size + rand_y

        uv_1d = y * self.img_w + x
        return uv_1d

        # uv = torch.zeros((num_samples, 2))

        # # y == uv_1d // img_w
        # uv[:, 1] = uv_1d // self.img_w
        # assert torch.max(uv[:, 1]) < self.img_h

        # uv[:, 0] = uv_1d - uv[:, 1] * self.img_w
        # assert torch.max(uv[:, 0]) < self.img_w

        # return uv
