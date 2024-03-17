import torch
import timeit

from ldm_patched.modules.tiled_upscale import (
    mask_step_mixed,
    mask_step_actual,
    mask_step_origin,
    mask_step_initial,
    mask_step_alternative,
)

# Initialize parameters
feather = 127
mask = torch.ones((1, 3, 512, 512))

# Create copies of the mask for testing
mask1 = mask.clone()
mask2 = mask.clone()

repeat = 10_000

# Time the original function
times_orig = timeit.repeat(lambda: mask_step_initial(feather,mask1.clone()), repeat=repeat, number=1)
print(f"Original function's fastest run took {min(times_orig)} seconds")

# Time the optimized function
times_opt = timeit.repeat(lambda: mask_step_actual(feather, mask2.clone()), repeat=repeat, number=1)
print(f"Optimized function's fastest run took {min(times_opt)} seconds")

# Calculate and print the relative difference in percentage
relative_diff = ((min(times_orig) - min(times_opt)) / min(times_orig)) * 100
print(f"The optimized function is {relative_diff:.2f}% faster than the original function")
