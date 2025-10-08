import os
from math import floor, ceil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional, Union, Tuple, List

def load_img(
        img_path: str,
        return_tensor: bool = True,
        dtype: Optional = torch.bfloat16,
):
    """
    Utility function to load an image from path

    :param img_path:
        (str) image path
    :param return_tensor:
        (bool) return torch.Tensor if True else numpy.ndarray.
        Defaults to True.
    :param dtype:
        (Optional) return dtype if return_tensor is True.
        Defaults to torch.bfloat16.
    :return: torch.Tensor if return_tensor is True else numpy.ndarray.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Please provide a valid LR image at {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img if not return_tensor else TF.to_tensor(img).to(dtype)


class PatchifyImage:
    def __init__(
            self,
            img: Union[torch.Tensor, np.ndarray],
            overlap: int = 0,
    ):
        """
        Patchify an image with an optional overlap.

        :param img:
            (torch.Tensor/np.ndarray) image to be patchified
        :param overlap:
            (int) Optional overlap in pixels.
            Defaults to 0.
        """
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        self.img = img

        self.overlap = overlap

    def pad_and_split_image(
            self,
            patch_size: Union[int, Tuple[int], List[int]],
    ):
        """
        Convert self.img to a `C` patches (tensor).

        :param patch_size:
            (int/tuple/list) patch size (excluding overlap) for patchify.
        :return:
            (torch.Tensor) patches of shape (C,...,patch_size+2*overlap,patch_size+2*overlap)
        """
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, list):
            assert len(patch_size) <= 2 and len(patch_size) > 0, \
                f"`patch_size` must be of type list of len 1 or 2 or tuple of len 1 or 2 or int, got list of len {len(patch_size)}"
            self.patch_size = tuple(patch_size) if len(patch_size) == 2 else (patch_size[0], patch_size[0])
        elif isinstance(patch_size, list):
            assert len(patch_size) <= 2 and len(patch_size) > 0, \
                f"`patch_size` must be of type list of len 1 or 2 or tuple of len 1 or 2 or int, got tuple of len {len(patch_size)}"
            self.patch_size = (patch_size[0], patch_size[0]) if len(patch_size) == 1 else patch_size
        else:
            raise TypeError(
                f"`patch_size` must be of type list of len 1 or 2 or tuple of len 1 or 2 or int, got {type(patch_size)}")
        padded_img = self.img
        if (self.patch_size[0] > (self.img.shape[-1])) or (self.patch_size[1] > (self.img.shape[-2])):
            self.constant_pad = True
            if self.overlap != 0:
                padded_img = F.pad(self.img, (self.overlap, self.overlap, self.overlap, self.overlap), mode='reflect')
            self.pad_t = 0
            self.pad_l = 0
            self.pad_b = (self.patch_size[0] - self.img.shape[-2])
            self.pad_r = (self.patch_size[1] - self.img.shape[-1])
            padded_img = F.pad(padded_img, (self.pad_l, self.pad_r, self.pad_t, self.pad_b), mode='constant', value=0)
            patches = padded_img.unsqueeze(0) # just 1 patch
            # self.num_patches = [1, 1]
            self.patches_shape = patches.shape
            return patches

        self.constant_pad = False

        pad_h = (self.patch_size[0] - (self.img.shape[-2] % self.patch_size[0])) % self.patch_size[0]
        pad_w = (self.patch_size[1] - (self.img.shape[-1] % self.patch_size[1])) % self.patch_size[1]

        self.pad_t, self.pad_l = floor(pad_h / 2) + self.overlap, floor(pad_w / 2) + self.overlap
        self.pad_b, self.pad_r = ceil(pad_h / 2) + self.overlap, ceil(pad_w / 2) + self.overlap

        padded_img = F.pad(self.img, (self.pad_l, self.pad_r, self.pad_t, self.pad_b), mode='reflect')

        patches = padded_img.unfold(-2, self.patch_size[0] + 2 * self.overlap, self.patch_size[0]) \
            .unfold(-2, self.patch_size[1] + 2 * self.overlap, self.patch_size[1])
        self.num_patches = patches.shape[-4:-2]
        patches = patches.flatten(-4, -3).transpose(-3, -4)
        self.patches_shape = patches.shape
        return patches

    def stitch_and_crop(
            self,
            patches: torch.Tensor,
            patch_size_scale: int,
    ):
        """
        Stitch image from patches and (optionally) crop.

        :param patches:
            (torch.Tensor) patches of shape (C,...,patch_size_scale*patch_size, patch_size_scale*patch_size)
            where C must be #patches returned by self.pad_and_split_image
            and patch_size as defined in self.pad_and_split_image.
        :param patch_size_scale:
            (int) scale parameter
        :return:
            (torch.Tensor) stitched and cropped image.
        """
        if self.constant_pad:
            patches = patches[..., : -(patch_size_scale * self.pad_b), : -(patch_size_scale * self.pad_r)]
            if self.overlap != 0:
                patches = patches[..., patch_size_scale*self.overlap:-patch_size_scale*self.overlap,\
                                        patch_size_scale*self.overlap:-patch_size_scale*self.overlap]
            return patches.squeeze(0)

        shape = self.img.shape[:-2] + self.num_patches + patches.shape[-2:]
        img = patches.transpose(-3, -4).view(shape)
        if self.overlap != 0:
            img = img[...,
                  patch_size_scale * self.overlap:-patch_size_scale * self.overlap, \
                  patch_size_scale * self.overlap:-patch_size_scale * self.overlap]
        img = img.transpose(-3, -2).flatten(-2, -1).flatten(-3, -2) \
              [..., patch_size_scale * (self.pad_t - self.overlap):-patch_size_scale * (self.pad_b - self.overlap), \
              patch_size_scale * (self.pad_l - self.overlap):-patch_size_scale * (self.pad_r - self.overlap)]
        return img

