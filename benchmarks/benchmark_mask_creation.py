import torch
import timeit


# Initialize parameters
feather = 127
mask = torch.ones((1, 3, 512, 512))

# Create copies of the mask for testing
mask1 = mask.clone()
mask2 = mask.clone()


def origin(feather, mask):
    for t in range(feather):
        mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
        mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
        mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
        mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))


def optimised(feather, mask):
    width, height = mask.shape[2], mask.shape[3]
    feather_ratio = (1.0 / feather) * (torch.arange(feather).float() + 1)
    view_1_1_minus_1_1 = 1, 1, -1, 1
    view_1_1_1_minus_1 = 1, 1, 1, -1
    mask[:, :, :feather, :] *= feather_ratio.view(view_1_1_minus_1_1)
    mask[:, :, :, :feather] *= feather_ratio.view(view_1_1_1_minus_1)
    mask[:, :, width - feather:, :] *= feather_ratio.flip(0).view(view_1_1_minus_1_1)
    mask[:, :, :, height - feather:] *= feather_ratio.flip(0).view(view_1_1_1_minus_1)


repeat = 100_000

# Time the original function
times_orig = timeit.repeat(lambda: origin(feather, mask1.clone()), repeat=repeat, number=1)
print(f"Original function's fastest run took {min(times_orig)} seconds")
print(f"Original function's average run took {sum(times_orig) / len(times_orig)} seconds")

# Time the optimized function
times_opt = timeit.repeat(lambda: optimised(feather, mask2.clone()), repeat=repeat, number=1)
print(f"Optimized function's fastest run took {min(times_opt)} seconds")
print(f"Optimized function's average run took {sum(times_opt) / len(times_opt)} seconds")

# Calculate and print the relative difference in percentage
relative_diff = ((min(times_orig) - min(times_opt)) / min(times_orig)) * 100
relative_average_diff = ((sum(times_opt) / len(times_opt) / len(times_orig)) / sum(times_orig) / len(times_orig)) * 100
print(f"The optimized function is {relative_diff:.2f}% faster than the original function")
print(f"The optimized function is {relative_diff:.2f}% faster on average than the original function")
