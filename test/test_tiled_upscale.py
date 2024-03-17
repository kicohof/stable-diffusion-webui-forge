import torch

from ldm_patched.modules.tiled_upscale import (
    mask_step_mixed,
    mask_step_actual,
    mask_step_origin,
    mask_step_initial,
    mask_step_alternative,
)


def test_mask_step_origin_vs_initial():
    # Initialize parameters
    feather = 127
    scale_ratio = 0.2  # replace this with your actual scale ratio
    inverted_feather = (1 / feather) * scale_ratio
    mask = torch.ones((1, 3, 512, 192))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and optimized functions
    mask_step_origin(feather, mask1)
    mask_step_initial(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_alternative():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 512, 192))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and optimized functions
    mask_step_origin(feather, mask1)
    mask_step_alternative(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_alternative_v2():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 192, 512))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and optimized functions
    mask_step_origin(feather, mask1)
    mask_step_alternative(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_mixed():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 512, 192))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and mixed functions
    mask_step_origin(feather, mask1)
    mask_step_mixed(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_mixed_v2():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 192, 512))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and mixed functions
    mask_step_origin(feather, mask1)
    mask_step_mixed(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_actual():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 512, 192))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and optimized functions
    mask_step_origin(feather, mask1)
    mask_step_actual(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)


def test_mask_step_origin_vs_actual_v2():
    # Initialize parameters
    feather = 127
    inverted_feather = (1 / feather)
    mask = torch.ones((1, 3, 192, 512))

    # Create copies of the mask for testing
    mask1 = mask.clone()
    mask2 = mask.clone()

    # Apply the original and optimized functions
    mask_step_origin(feather, mask1)
    mask_step_actual(feather, mask2)

    # Assert if the outputs are the same
    assert torch.allclose(mask1, mask2)
