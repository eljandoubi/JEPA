
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
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)

        return h, w 

    def _sample_block_mask(self, b_size: tuple[int, int], acceptable_regions: list[torch.Tensor]|None=None):
        """sample a rectangular block of shape (h,w)"""
        h, w = b_size
        tries = 0
        timeout = og_timeout = 25
        valid_mask = False
        while not valid_mask:
            # sample a random top left corner
            top = torch.randint(0, self.height - h, (1,)).item()
            left = torch.randint(0, self.width - w, (1,)).item()

            # create a mask of zeros
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)

            # set the block to True
            mask[top:top+h, left:left+w] = 1

            if acceptable_regions is not None:
                # check if the block overlaps with any acceptable region
                N = max(len(acceptable_regions)-tries, 0)  # check at most 10 regions
                for i in range(N):
                    mask *= acceptable_regions[i]

            mask_ids = torch.nonzero(mask.float(), as_tuple=False).squeeze(1)
            valid_mask = len(mask_ids) > self.min_keep

            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    timeout = og_timeout

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0

        return mask_ids, mask_complement