
import math

import torch


class JEPAMaskSampler:
    def __init__(self, 
                 image_size: int=224,
                 patch_size: int=16,
                 enc_mask_scale: tuple[float, float]=(0.85, 1.0),
                 pred_mask_scale: tuple[float, float]=(0.15, 0.2),
                 aspect_ratio: tuple[float, float]=(0.75, 1.5),
                 nenc: int = 1,
                 npred: int = 4,
                 min_keep: int = 4,
                 allow_overlap: bool = False
                 ):
        self.height = self.width = image_size//patch_size
        self.num_patches = self.height * self.width
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap


    def _sample_block_size(
        self,
        scale: tuple[float, float],
        aspect_ratio_scale: tuple[float, float]
    ):
        
        """sample (h,w) for one block"""
        
        ### select a random proportion between 0 and 1
        rand = torch.rand(1).item() # [0,1]

        ### unpack min and max size from scale
        min_s, max_s = scale

        ### random sample some size between these two 
        mask_scale = min_s + rand * (max_s - min_s) # some val between min_s,max_s
        max_keep = int(self.height * self.width * mask_scale) # num patches to keep

        min_ar, max_ar = aspect_ratio_scale 
        aspect_ratio = min_ar + rand * (max_ar - min_ar) # some val between min_ar, max_ar

        ### get (h,w) that has max_keep elements inside w/ the wanted aspect ratio
        h = round(math.sqrt(max_keep * aspect_ratio))
        w = round(math.sqrt(max_keep / aspect_ratio))
 
        # Clamp to grid
        while h >= self.height: h -= 1
        while w >= self.width:  w -= 1

        return h, w 