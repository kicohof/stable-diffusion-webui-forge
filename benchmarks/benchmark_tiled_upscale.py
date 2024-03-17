import torch
import timeit

from ldm_patched.modules.tiled_upscale import (
    mask_step_actual,
    mask_step_initial,
)

# Initialize parameters
feather = 127
mask = torch.ones((1, 3, 512, 512))

# Create copies of the mask for testing
mask1 = mask.clone()
mask2 = mask.clone()

repeat = 100_000

# Time the original function
times_orig = timeit.repeat(lambda: mask_step_initial(feather, mask1.clone()), repeat=repeat, number=1)
print(f"Original function's fastest run took {min(times_orig)} seconds")
print(f"Original function's average run took {sum(times_orig) / len(times_orig)} seconds")

# Time the optimized function
times_opt = timeit.repeat(lambda: mask_step_actual(feather, mask2.clone()), repeat=repeat, number=1)
print(f"Optimized function's fastest run took {min(times_opt)} seconds")
print(f"Optimized function's average run took {sum(times_opt) / len(times_opt)} seconds")

# Calculate and print the relative difference in percentage
relative_diff = ((min(times_orig) - min(times_opt)) / min(times_orig)) * 100
relative_average_diff = ((sum(times_opt) / len(times_opt) / len(times_orig)) / sum(times_orig) / len(times_orig)) * 100
print(f"The optimized function is {relative_diff:.2f}% faster than the original function")
print(f"The optimized function is {relative_diff:.2f}% faster on average than the original function")

# Result with repeat = 10_000
# Original function's fastest run took 0.0157868000005692 seconds
# Original function's average run took 0.019530934530010563 seconds
# Optimized function's fastest run took 0.00023919999875943176 seconds
# Optimized function's fastest run took 0.00034369515998914724 seconds
# The optimized function is 98.48% faster than the original function
# The optimized function is 98.48% faster on average than the original function

# Result with repeat = 100_000
# Original function's fastest run took 0.01545079999959853 seconds
# Original function's average run took 0.018991429873002872 seconds
# Optimized function's fastest run took 0.0002224000018031802 seconds
# Optimized function's average run took 0.0003363384599928031 seconds
# The optimized function is 98.56% faster than the original function
# The optimized function is 98.56% faster on average than the original function

