import torch

from collections import deque
from itertools import product


@torch.inference_mode
def tiled_scale_optimized(samples, function, tile_x=64, tile_y=64,
                          overlap=8, upscale_amount=4, out_channels=3,
                          output_device="cpu", pbar=None):

    def scale(value):
        return round(value * upscale_amount)

    def tensor_dim(ba, hi, wi):
        return ba, out_channels, scale(wi), scale(hi)

    def overlap_range(value, tile, _overlap_=overlap):
        return range(0, value, tile - _overlap_)

    def clamp(val, min_val=0, max_val=1):
        return max(min_val, min(max_val - overlap, val))

    def masking(dimensions):
        y = clamp(dimensions[0], max_val=s_width)
        x = clamp(dimensions[1], max_val=s_height)
        s_int = s[:, :, y:y + tile_y, x:x + tile_x]
        ps = function(s_int).to(output_device)
        mask = create_mask(ps)
        out[:, :, scale(y):scale(y + tile_y), scale(x):scale(x + tile_x)] += ps * mask
        out_div[:, :, scale(y):scale(y + tile_y), scale(x):scale(x + tile_x)] += mask

        if pbar is not None:
            pbar.update(1)

        pass

    def create_mask(ps):
        mask = torch.ones_like(ps)
        mask_width, mask_height = mask.shape[2], mask.shape[3]
        feather = scale(overlap)
        feather_ratio = (1.0 / feather) * (torch.arange(feather).float() + 1)
        mask[:, :, :feather, :] *= feather_ratio.view(1, 1, -1, 1)
        mask[:, :, :, :feather] *= feather_ratio.view(1, 1, 1, -1)
        mask[:, :, mask_width - feather:, :] *= feather_ratio.flip(0).view(1, 1, -1, 1)
        mask[:, :, :, mask_height - feather:] *= feather_ratio.flip(0).view(1, 1, 1, -1)
        return mask

    batch, _, height, width = samples.shape
    output = torch.empty(tensor_dim(batch, width, height), device=output_device)

    for b in range(batch):
        s = samples[b:b + 1]
        s_batch, _, s_width, s_height = s.shape
        out = torch.zeros(tensor_dim(s_batch, s_height, s_width), device=output_device)
        out_div = out.detach().clone()
        consume(map(masking, product(overlap_range(s_width, tile_y), overlap_range(s_height, tile_x))))
        output[b:b + 1] = out / out_div

    return output


def consume(generator):
    deque(generator, maxlen=0)
