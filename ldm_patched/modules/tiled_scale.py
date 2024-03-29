import torch


@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device=output_device)
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device=output_device)
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                x = max(0, min(s.shape[-1] - overlap, x))
                y = max(0, min(s.shape[-2] - overlap, y))
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                ps = function(s_in).to(output_device)
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                width, height = mask.shape[2], mask.shape[3]
                feather_ratio = (1.0 / feather) * (torch.arange(feather).float() + 1)
                view_1_1_minus_1_1 = 1, 1, -1, 1
                view_1_1_1_minus_1 = 1, 1, 1, -1
                mask[:, :, :feather, :] *= feather_ratio.view(view_1_1_minus_1_1)
                mask[:, :, :, :feather] *= feather_ratio.view(view_1_1_1_minus_1)
                mask[:, :, width - feather:, :] *= feather_ratio.flip(0).view(view_1_1_minus_1_1)
                mask[:, :, :, height - feather:] *= feather_ratio.flip(0).view(view_1_1_1_minus_1)

                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b+1] = out/out_div
    return output
