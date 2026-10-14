
import math

import torch


class JEPAMaskSampler:
    """A class to sample masks for JEPA training."""
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
        """Initialize the mask sampler."""
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
        
        """sample (h,w) for one block
        Args:
            scale: (min,max) proportion of the image to keep
            aspect_ratio_scale: (min,max) aspect ratio of the block
        Returns:
            (h,w): height and width of the block
        """
        
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
        """sample a rectangular block of shape (h,w)
        Args:
            b_size: (h,w) of the block to sample
            acceptable_regions: list of binary masks of shape (H,W) that indicate where the block can be placed
        Returns:
            mask_ids: indices of the masked patches
            mask_complement: binary mask of shape (H,W) where 1 indicates the patch is kept and 0 indicates the patch is masked
        """
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

            mask_ids = torch.nonzero(mask.flatten()).squeeze(1)
            valid_mask = len(mask_ids) > self.min_keep

            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0

        return mask_ids, mask_complement

    def sample_masks(self, batch_size: int):
        """sample masks for a batch of images
        Args:
            batch_size: number of images in the batch
        Returns:
            context_ids: indices of the context patches (B, nenc, Nc)
            target_ids: indices of the target patches (B, npred, Nt)
        """

        p_size = self._sample_block_size(self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(self.enc_mask_scale, (1., 1.)) # enc mask is always square
        collated_masks_pred, collated_masks_enc = [], []

        min_keep_pred = self.num_patches
        min_keep_enc = self.num_patches

        for _ in range(batch_size):
            masks_P, masks_Cs = [], []
            for _ in range(self.npred):
                mask_ids, mask_complement = self._sample_block_mask(p_size)
                masks_P.append(mask_ids)
                masks_Cs.append(mask_complement)
                min_keep_pred = min(min_keep_pred, len(mask_ids))


            collated_masks_pred.append(masks_P)

            masks_E = []
            for _ in range(self.nenc):
                acceptable_regions = masks_Cs if not self.allow_overlap else None
                mask_ids, _ = self._sample_block_mask(e_size, acceptable_regions)
                masks_E.append(mask_ids)
                min_keep_enc = min(min_keep_enc, len(mask_ids))

            collated_masks_enc.append(masks_E)

        ### stack it all up
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_enc = [[cm[:min_keep_enc]  for cm in cm_list] for cm_list in collated_masks_enc]
 
        target_ids = torch.stack([torch.stack(m) for m in collated_masks_pred])  # (B, npred, Nt)
        context_ids = torch.stack([torch.stack(m) for m in collated_masks_enc])   # (B, nenc,  Nc)
 
        return context_ids, target_ids


if __name__ == "__main__":
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
 
    def plot_masks(
        context_ids,
        target_ids,
        target_blocks,  
        grid_size=14,
        title="I-JEPA Masking Pattern",
        sample_idx=0
    ):
        context_ids = np.asarray(context_ids).flatten()
        target_ids  = np.unique(np.asarray(target_ids).flatten())
 
        assert len(set(context_ids.tolist()) & set(target_ids.tolist())) == 0, \
            "Context and target overlap!"
 
        # 0=unused  1=context  2=target
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        for idx in context_ids:
            r, c = divmod(int(idx), grid_size)
            grid[r, c] = 1
        for idx in target_ids:
            r, c = divmod(int(idx), grid_size)
            grid[r, c] = 2
 
        colors = ["#1a1a2e", "#4a9eff", "#ff4d6d"]
        labels = ["Unused", "Context", "Target"]
        cmap   = plt.matplotlib.colors.ListedColormap(colors)  # type: ignore
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm   = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N) # type: ignore
 
        _, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d1a")
        ax.set_facecolor("#0d0d1a")
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
 
        # Subtle grid lines
        for x in range(grid_size + 1):
            ax.axhline(x - 0.5, color="#ffffff22", linewidth=0.8)
            ax.axvline(x - 0.5, color="#ffffff22", linewidth=0.8)
 
        block_colors = ["#ffd166", "#06d6a0", "#f4a261", "#c77dff"]
        for b_idx, block_ids in enumerate(target_blocks):
            block_ids = np.asarray(block_ids).flatten()
            rows = [int(idx) // grid_size for idx in block_ids]
            cols = [int(idx) %  grid_size for idx in block_ids]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            color = block_colors[b_idx % len(block_colors)]
 
            rect = mpatches.FancyBboxPatch(
                (c_min - 0.5, r_min - 0.5),
                c_max - c_min + 1,
                r_max - r_min + 1,
                boxstyle="round,pad=0.05",
                linewidth=5.0,
                edgecolor=color,
                facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect)
 
            # Small label in top-left corner of each box
            ax.text(c_min - 0.35, r_min - 0.35, f"T{b_idx+1}",
                    color=color, fontsize=8, fontfamily="monospace",
                    fontweight="bold", va="top", zorder=6)
 
        n_ctx = (grid == 1).sum()
        n_tgt = (grid == 2).sum()
        ax.set_title(
            f"{title} — Sample {sample_idx+1}\n"
            f"Context: {n_ctx} patches    Target: {n_tgt} patches",
            color="white", fontsize=13, pad=16, fontfamily="monospace"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
 
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f"{labels[i]} ({(grid==i).sum()})")
            for i in range(3) if (grid == i).any()
        ] + [
            mpatches.Patch(edgecolor=block_colors[b % len(block_colors)],
                           facecolor="none", linewidth=2,
                           label=f"Block T{b+1}")
            for b in range(len(target_blocks))
        ]
        ax.legend(handles=legend_patches, loc="upper right", framealpha=0.35,
                  facecolor="#1a1a2e", edgecolor="#ffffff30",
                  labelcolor="white", fontsize=9, prop={"family": "monospace"})
 
        plt.tight_layout()
        plt.show()
 
    sampler = JEPAMaskSampler(
        image_size=224, patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        nenc=1, npred=4,
    )
 
    ctx, tgt = sampler.sample_masks(batch_size=4)

 
    ctx_flat = ctx.flatten(1)
    tgt_flat = tgt.flatten(1) 

    print("Context shape:", ctx_flat.shape, "Target shape:", tgt_flat.shape)
    print("Intersection (should be empty):",
          set(ctx_flat[0].tolist()) & set(tgt_flat[0].tolist()))
 
    for i in range(len(ctx_flat)):
        plot_masks(
            context_ids=ctx_flat[i].numpy(),
            target_ids=tgt_flat[i].numpy(),
            target_blocks=tgt[i].numpy(),   # (npred, Nt)
            grid_size=14,
            title="I-JEPA Masking Pattern",
            sample_idx=i,
        )