import os
import torch
import pickle
import pytest
import requests
import itertools
import numpy as np
from ldm_patched.modules.utils import (
    ProgressBar,
    tiled_scale,
    get_tiled_scale_steps,
)


def test_options_write(base_url):
    url_options = f"{base_url}/sdapi/v1/options"
    response = requests.get(url_options)
    assert response.status_code == 200

    pre_value = response.json()["send_seed"]

    assert requests.post(url_options, json={'send_seed': (not pre_value)}).status_code == 200

    response = requests.get(url_options)
    assert response.status_code == 200
    assert response.json()['send_seed'] == (not pre_value)

    requests.post(url_options, json={"send_seed": pre_value})


@pytest.mark.parametrize("url", [
    "sdapi/v1/cmd-flags",
    "sdapi/v1/samplers",
    "sdapi/v1/upscalers",
    "sdapi/v1/sd-models",
    "sdapi/v1/hypernetworks",
    "sdapi/v1/face-restorers",
    "sdapi/v1/realesrgan-models",
    "sdapi/v1/prompt-styles",
    "sdapi/v1/embeddings",
])
def test_get_api_url(base_url, url):
    assert requests.get(f"{base_url}/{url}").status_code == 200


def test_tiled_scale():

    # print(type(samples), str(samples), repr(samples), sep='\n')
    # print(type(tile_x), str(tile_x), repr(tile_x), sep='\n')
    # print(type(tile_y), str(tile_y), repr(tile_y), sep='\n')
    # print(type(overlap), str(overlap), repr(overlap), sep='\n')
    # print(type(upscale_amount), str(upscale_amount), repr(upscale_amount), sep='\n')
    # print(type(out_channels), str(out_channels), repr(out_channels), sep='\n')
    # print(type(output_device), str(output_device), repr(output_device), sep='\n')
    # print(type(pbar), str(pbar), repr(pbar), sep='\n')

    # <class 'torch.Tensor'>
    # tensor([[[[ 10.6564,  13.5948,  13.1620,  ...,  -6.4845,  -7.8855,  -8.3852],
    #           [ 14.7464,   9.3423,   9.8401,  ...,  -6.3528,  -7.3278, -11.2179],
    #           [ 11.1277,   7.6965,   8.9189,  ...,  -8.4239,  -5.1113,  -8.9681],
    #           ...,
    #           [ -8.5718,  -7.6852,  -9.4906,  ...,   6.5628,   6.4846,   3.9870],
    #           [ -7.6421,  -8.5469,  -9.2460,  ...,   2.7505,   3.2628,   2.7579],
    #           [ -9.9144,  -8.5709,  -7.4814,  ...,   3.5249,   0.7133,   4.4951]],
    #
    #          [[ -4.1922,   2.6221,   2.3272,  ...,  -3.0273,  -7.2848,  -4.1913],
    #           [  4.0396,   0.2088,  -1.0225,  ...,  -1.1684,  -4.0181,  -5.7886],
    #           [ -1.1910,  -1.8876,   1.4097,  ...,  -8.7865,  -2.2094,  -3.7080],
    #           ...,
    #           [  1.5121,   4.9144,   3.1341,  ...,   9.0333,   8.4236,   4.2322],
    #           [  7.8803,   7.9532,   4.5404,  ...,   1.7923,   4.0771,   6.8867],
    #           [  0.8991,   6.8746,   6.9973,  ...,   4.9107,   1.4256,   6.5995]],
    #
    #          [[  1.9748,  -2.7250,  -2.9876,  ...,   1.6424,   6.1266,  -4.3527],
    #           [  2.8533,   7.5316,   7.6706,  ...,   1.6738,  -6.5573,  -1.4925],
    #           [ -2.6884,   3.1627,   0.1662,  ...,  -2.4023,  -8.4415,  -3.4184],
    #           ...,
    #           [ -2.8455,   4.1061,   8.1831,  ...,   0.7920,  -2.4840,   0.9210],
    #           [ -4.9119,  -2.6098,   5.3879,  ...,  -8.4641,  -9.1992,  -6.0148],
    #           [  0.7606,   0.2847,   5.6848,  ...,   3.3726,  -2.5544,   1.2803]],
    #
    #          [[ -5.8247,  -3.8514,  -4.1890,  ...,   1.8840,  -2.9391,  -1.1333],
    #           [ -0.3188,  -5.6875,  -5.0217,  ...,  -4.8680,  -3.9451,  -8.0427],
    #           [ -4.1480,  -9.4284,  -5.4854,  ...,  -7.0970,  -4.3694,  -8.1360],
    #           ...,
    #           [  2.2535,   5.5329,   1.9146,  ...,   1.6541,   7.3823,   5.8807],
    #           [  0.5108,   0.8826,   5.5405,  ...,   1.8384,   0.7730,  -4.6430],
    #           [ -1.7819,  -1.7637,   1.5558,  ...,   0.6422,  -3.3172,   1.5595]]]],
    #        device='cuda:0')
    # tensor([[[[ 10.6564,  13.5948,  13.1620,  ...,  -6.4845,  -7.8855,  -8.3852],
    #           [ 14.7464,   9.3423,   9.8401,  ...,  -6.3528,  -7.3278, -11.2179],
    #           [ 11.1277,   7.6965,   8.9189,  ...,  -8.4239,  -5.1113,  -8.9681],
    #           ...,
    #           [ -8.5718,  -7.6852,  -9.4906,  ...,   6.5628,   6.4846,   3.9870],
    #           [ -7.6421,  -8.5469,  -9.2460,  ...,   2.7505,   3.2628,   2.7579],
    #           [ -9.9144,  -8.5709,  -7.4814,  ...,   3.5249,   0.7133,   4.4951]],
    #
    #          [[ -4.1922,   2.6221,   2.3272,  ...,  -3.0273,  -7.2848,  -4.1913],
    #           [  4.0396,   0.2088,  -1.0225,  ...,  -1.1684,  -4.0181,  -5.7886],
    #           [ -1.1910,  -1.8876,   1.4097,  ...,  -8.7865,  -2.2094,  -3.7080],
    #           ...,
    #           [  1.5121,   4.9144,   3.1341,  ...,   9.0333,   8.4236,   4.2322],
    #           [  7.8803,   7.9532,   4.5404,  ...,   1.7923,   4.0771,   6.8867],
    #           [  0.8991,   6.8746,   6.9973,  ...,   4.9107,   1.4256,   6.5995]],
    #
    #          [[  1.9748,  -2.7250,  -2.9876,  ...,   1.6424,   6.1266,  -4.3527],
    #           [  2.8533,   7.5316,   7.6706,  ...,   1.6738,  -6.5573,  -1.4925],
    #           [ -2.6884,   3.1627,   0.1662,  ...,  -2.4023,  -8.4415,  -3.4184],
    #           ...,
    #           [ -2.8455,   4.1061,   8.1831,  ...,   0.7920,  -2.4840,   0.9210],
    #           [ -4.9119,  -2.6098,   5.3879,  ...,  -8.4641,  -9.1992,  -6.0148],
    #           [  0.7606,   0.2847,   5.6848,  ...,   3.3726,  -2.5544,   1.2803]],
    #
    #          [[ -5.8247,  -3.8514,  -4.1890,  ...,   1.8840,  -2.9391,  -1.1333],
    #           [ -0.3188,  -5.6875,  -5.0217,  ...,  -4.8680,  -3.9451,  -8.0427],
    #           [ -4.1480,  -9.4284,  -5.4854,  ...,  -7.0970,  -4.3694,  -8.1360],
    #           ...,
    #           [  2.2535,   5.5329,   1.9146,  ...,   1.6541,   7.3823,   5.8807],
    #           [  0.5108,   0.8826,   5.5405,  ...,   1.8384,   0.7730,  -4.6430],
    #           [ -1.7819,  -1.7637,   1.5558,  ...,   0.6422,  -3.3172,   1.5595]]]],
    #        device='cuda:0')
    # <class 'int'>
    # 64
    # 64
    # <class 'int'>
    # 64
    # 64
    # <class 'int'>
    # 16
    # 16
    # <class 'int'>
    # 8
    # 8
    # <class 'int'>
    # 3
    # 3
    # <class 'torch.device'>
    # cpu
    # device(type='cpu')
    # <class 'ldm_patched.modules.utils.ProgressBar'>
    # <ldm_patched.modules.utils.ProgressBar object at 0x000001E42C9443D0>
    # <ldm_patched.modules.utils.ProgressBar object at 0x000001E42C9443D0>

    # self.vae_dtype
    # <class 'torch.dtype'>
    # torch.bfloat16
    # torch.bfloat16
    # self.device
    # <class 'torch.device'>
    # cuda:0
    # device(type='cuda', index=0)

    module_dir = os.path.dirname(__file__)

    with open(os.path.join(module_dir, 'test_files', 'samples.pth'), 'rb') as samples_file:
        samples = pickle.load(samples_file)

    with open(os.path.join(module_dir, 'test_files', 'first_stage_model.pth'), 'rb') as first_stage_model_file:
        first_stage_model = pickle.load(first_stage_model_file)

    vae_dtype = torch.bfloat16
    device = torch.device(type="cuda", index=0)

    def decode_function(a):
        a_tensor = a.to(vae_dtype).to(device)
        return (first_stage_model.decode(a_tensor) + 1.0).float()

    tile_x = 64
    tile_y = 64
    overlap = 16

    sample_shape_0, sample_shape_2, sample_shape_3 = samples.shape[0], samples.shape[2], samples.shape[3]
    steps_part1 = get_tiled_scale_steps(sample_shape_3, sample_shape_2, tile_x, tile_y, overlap)
    steps_part2 = get_tiled_scale_steps(sample_shape_3, sample_shape_2, tile_x // 2, tile_y * 2, overlap)
    steps_part3 = get_tiled_scale_steps(sample_shape_3, sample_shape_2, tile_x * 2, tile_y // 2, overlap)
    steps = sample_shape_0 * (steps_part1 + steps_part2 + steps_part3)
    pbar = ProgressBar(steps, title='VAE tiled decode')

    tiled_scale(
        samples=samples,
        function=decode_function,
        tile_x=tile_x * 2,
        tile_y=tile_y // 2,
        overlap=16,
        upscale_amount=8,
        out_channels=3,
        output_device="cpu",
        pbar=pbar
    )

    tiled_scale(
        samples=samples,
        function=decode_function,
        tile_x=tile_x // 2,
        tile_y=tile_y * 2,
        overlap=16,
        upscale_amount=8,
        out_channels=3,
        output_device="cpu",
        pbar=pbar
    )

    tiled_scale(
        samples=samples,
        function=decode_function,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=16,
        upscale_amount=8,
        out_channels=3,
        output_device="cpu",
        pbar=pbar
    )


def test_mask_step_simplification():

    # Original function
    def mask_step_1(feather_ratio, mask, width, height, t):
        mask[:, :, t:t + 1, :] *= feather_ratio
        mask[:, :, :, t:t + 1] *= feather_ratio
        mask[:, :, width - 1 - t:width - t, :] *= feather_ratio
        mask[:, :, :, height - 1 - t:height - t] *= feather_ratio
        return mask

    def mask_step_2(feather_ratio, mask, width, height, t):
        mask[:, :, np.array([t, width - 1 - t]), :] *= feather_ratio
        mask[:, :, :, np.array([t, height - 1 - t])] *= feather_ratio
        return mask

    # Test both functions
    feather_ratio = 0.337
    width = 6
    height = 4
    t = 2
    mask_1 = np.ones((1, 3, width, height), dtype=np.float32)
    mask_2 = np.ones((1, 3, width, height), dtype=np.float32)
    assert np.allclose(mask_1, mask_2)

    assert np.allclose(mask_step_1(feather_ratio, mask_1, width, height, t),
                       mask_step_2(feather_ratio, mask_2, width, height, t))


def test_range():
    y_range = range(64)
    x_range = range(64)

    print('for-loop:')
    for y in y_range:
        for x in x_range:
            print('x=', x, 'y=', y)

    print('product-for-loop:')
    for y, x in itertools.product(y_range, x_range):
        print('x=', x, 'y=', y)



