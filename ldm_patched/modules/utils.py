# 1st edit https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge


import torch
import math
import struct
import itertools
import numpy as np
import safetensors.torch
import ldm_patched.modules.checkpoint_pickle

from PIL import Image
from tqdm import tqdm
from functools import cache
from collections import deque
from dataclasses import dataclass


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=ldm_patched.modules.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)

def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params

def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict

def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
    return sd

UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}

def unet_to_diffusers(unet_config):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map

def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor

def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output

def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict

def safetensors_header(safetensors_path, max_size=100*1024*1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack('<Q', header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)

def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))
    del prev

def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''
        
        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5] 
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res
    
    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)
        
        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)
    
    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def common_upscale(samples, width, height, upscale_method, crop):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, width, height)
    elif upscale_method == "lanczos":
        return lanczos(s, width, height)
    else:
        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@dataclass(repr=False, eq=False, match_args=False)
class TilingData:
    samples: torch.Tensor
    sample_shape_0: int
    sample_shape_2: int
    sample_shape_3: int
    decode_function: callable
    tile_x: int
    tile_y: int
    overlap: int
    upscale_amount: int
    out_channels: int
    output_device: str
    pbar: 'ProgressBar'
    feather: float
    inverted_feather: float

    def update_progressbar(self, step):
        if self.pbar is not None:
            self.pbar.update(step)
        pass

    pass

    @cache
    def forward_samples(self, b):
        self.samples = self.samples[b:b + 1]
        self.sample_shape_0, self.sample_shape_2, self.sample_shape_3 = self.samples.shape[0], self.samples.shape[2], \
            self.samples.shape[3]
        return self.samples

    @cache
    def out_dimensions(self):
        return self.sample_shape_0, self.out_channels, round(self.sample_shape_2 * self.upscale_amount), \
            round(self.sample_shape_3 * self.upscale_amount)

    @cache
    def x_y_ranges(self):
        return range(0, self.sample_shape_2, self.tile_y - self.overlap), \
            range(0, self.sample_shape_3, self.tile_x - self.overlap)

    @cache
    def scale(self, value):
        return round(value * self.upscale_amount)

    @cache
    def tile_scale(self, value, tile):
        return round((value + tile) * self.upscale_amount)

    pass


def exhaust(generator):
    any(generator)


@torch.inference_mode()
def tiled_scale(
        samples,
        function,
        tile_x=64, tile_y=64,
        overlap=8,
        upscale_amount=4,
        out_channels=3,
        output_device="cpu",
        pbar=None
):

    sample_shape_0, sample_shape_2, sample_shape_3 = samples.shape[0], samples.shape[2], samples.shape[3]
    feather = round(overlap * upscale_amount)
    inverted_feather = (1.0 / feather)

    tiling_data = TilingData(
        samples=samples,
        sample_shape_0=sample_shape_0,
        sample_shape_2=sample_shape_2,
        sample_shape_3=sample_shape_3,
        decode_function=function,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        pbar=pbar,
        feather=feather,
        inverted_feather=inverted_feather
    )

    output = torch.empty(tiling_data.out_dimensions(), device=output_device)

    exhaust(
        (tiled_scale_step(b, tiling_data, output)
         for b in range(tiling_data.sample_shape_0))
    )

    return output


def tiled_scale_step(b, tiling_data, output):
    out, out_div = tiled_scale_step_(b, tiling_data)
    output[b:b + 1] = out / out_div
    pass


def tiled_scale_step_(b, tiling_data):

    tiling_data.forward_samples(b)
    out_dimensions = tiling_data.out_dimensions()
    out = torch.zeros(out_dimensions, device=tiling_data.output_device)
    out_div = out.detach().clone()

    sample = tiling_data.samples
    overlap_x, overlap_y = sample.shape[-1] - tiling_data.overlap, sample.shape[-2] - tiling_data.overlap

    exhaust(
        (scale_step(tiling_data, out, out_div, overlap_x, overlap_y, x, y)
         for y, x in itertools.product(*tiling_data.x_y_ranges()))
    )

    return out, out_div


def scale_step(
        tiling_data,
        out, out_div,
        overlap_x, overlap_y,
        x_coord, y_coord
):
    x, y = max(0, min(overlap_x, x_coord)), max(0, min(overlap_y, y_coord))
    sample_input = tiling_data.samples[:, :, y:y+tiling_data.tile_y, x:x+tiling_data.tile_x]
    decoded_sample = tiling_data.decode_function(sample_input).to(tiling_data.output_device)
    mask = torch.ones_like(decoded_sample)  # (1, 3, 192, 192), (1, 3, 192, 512), (1, 3, 512, 192), (1, 3, 512, 512)

    exhaust(
        (mask_step((tiling_data.inverted_feather * (t + 1)), mask, mask.shape[2], mask.shape[3], t)
         for t in range(tiling_data.feather))
    )

    y_scale, x_scale = tiling_data.scale(y), tiling_data.scale(x)
    y_tile_scale, x_tile_scale = tiling_data.tile_scale(y, tiling_data.tile_y), \
        tiling_data.tile_scale(x, tiling_data.tile_x)

    out[:, :, y_scale:y_tile_scale, x_scale:x_tile_scale] += decoded_sample * mask
    out_div[:, :, y_scale:y_tile_scale, x_scale:x_tile_scale] += mask

    tiling_data.update_progressbar(1)
    pass


def mask_step(feather_ratio, mask, width, height, t):
    mask[:, :, t:t+1, :] *= feather_ratio
    mask[:, :, :, t:t+1] *= feather_ratio
    mask[:, :, width - 1 - t:width - t, :] *= feather_ratio
    mask[:, :, :, height - 1 - t:height - t] *= feather_ratio
    pass


PROGRESS_BAR_ENABLED = True


def set_progress_bar_enabled(enabled):
    global PROGRESS_BAR_ENABLED
    PROGRESS_BAR_ENABLED = enabled


PROGRESS_BAR_HOOK = None


def set_progress_bar_global_hook(function):
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function


class ProgressBar:
    def __init__(self, total, title=None):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK
        self.tqdm = tqdm(total=total, desc=title)

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        inc = value - self.current
        self.tqdm.update(inc)
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview)
        if self.current >= self.total:
            self.tqdm.close()

    def update(self, value):
        self.update_absolute(self.current + value)
