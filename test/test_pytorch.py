import numpy as np
import torch


def test_vector_slicing():
    k = torch.Tensor([2, 3, 3])
    t = torch.Tensor([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3]])

    h, w = t.shape

    # create a mask of ones with size h, w+1
    mask = torch.ones(h, w + 1)

    # set other elements to 0, depending on length of each row
    mask[torch.arange(h), k.long()] = 0.
    mask = mask.cumprod(dim=1)

    # multiply the mask
    actual = t * mask[:, :-1]
    print()
    print(actual)
    expected = torch.Tensor([[1., 0., 0.],
                             [2., 2., 2.],
                             [3., 3., 0.]])

    assert torch.allclose(actual, expected)
    pass


def test_diagonal():

    feather = 4
    diagonal_vector = torch.arange(1, feather + 1, dtype=torch.bfloat16, device='cpu')
    a = torch.zeros((feather, feather), dtype=torch.bfloat16, device='cpu')
    print(diagonal_vector)
    print(a)

    ind = np.diag_indices(a.shape[0])
    a[ind[0], ind[1]] = diagonal_vector
    print(a)
    pass


def test_diagonal_filling():
    feather = 4
    n = 8
    m = 8
    mask = torch.zeros((n, m), dtype=torch.bfloat16, device='cpu')
    diagonal_vector = torch.zeros(1, mask.shape[0], dtype=torch.bfloat16, device='cpu')
    diagonal_vector[0, 0:feather] = torch.arange(1, feather + 1, dtype=torch.bfloat16, device='cpu')

    print()
    print(diagonal_vector)
    print(mask)

    ind = np.diag_indices(mask.shape[0])
    mask[ind[0], ind[1]] = diagonal_vector
    print(mask)
    pass


def test_asymmetric_diagonal_filling():
    feather = 4
    n = 6
    m = 8
    mask = torch.ones((1, 3, n, m), dtype=torch.bfloat16, device='cpu')
    # noinspection DuplicatedCode
    diagonal_vector = torch.ones(1, mask.shape[2], dtype=torch.bfloat16, device='cpu')
    diagonal_vector[0, 0:feather] = torch.arange(1, feather + 1, dtype=torch.bfloat16, device='cpu') * 0.0078

    print()
    print(diagonal_vector)
    print(mask)

    ind = np.diag_indices(mask.shape[2])
    mask[:, :, ind[0], ind[1]] = diagonal_vector
    print(mask)
    pass
