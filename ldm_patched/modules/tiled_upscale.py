import torch


def mask_step_actual(feather: int, mask: torch.Tensor) -> None:
    width, height = mask.shape[2], mask.shape[3]
    feather_ratio = (1.0 / feather) * (torch.arange(feather).float() + 1)
    view_1_1_minus_1_1 = 1, 1, -1, 1
    view_1_1_1_minus_1 = 1, 1, 1, -1

    mask[:, :, :feather, :] *= feather_ratio.view(view_1_1_minus_1_1)
    mask[:, :, :, :feather] *= feather_ratio.view(view_1_1_1_minus_1)
    mask[:, :, width-feather:, :] *= feather_ratio.flip(0).view(view_1_1_minus_1_1)
    mask[:, :, :, height-feather:] *= feather_ratio.flip(0).view(view_1_1_1_minus_1)
    pass


def mask_step_initial(feather: int, mask: torch.Tensor):
    for t in range(feather):
        mask[:, :, t:1 + t, :] *= ((1.0 / feather) * (t + 1))
        mask[:, :, mask.shape[2] - 1 - t: mask.shape[2] - t, :] *= ((1.0 / feather) * (t + 1))
        mask[:, :, :, t:1 + t] *= ((1.0 / feather) * (t + 1))
        mask[:, :, :, mask.shape[3] - 1 - t: mask.shape[3] - t] *= ((1.0 / feather) * (t + 1))
    pass


def mask_step_origin(feather: int, mask: torch.Tensor) -> None:
    width, height = mask.shape[2], mask.shape[3]
    inverted_feather = (1.0 / feather)

    for t in range(feather):
        feather_ratio = inverted_feather * (t + 1)
        mask[:, :, t:1 + t, :] *= feather_ratio
        mask[:, :, width - 1 - t: width - t, :] *= feather_ratio
        mask[:, :, :, t:1 + t] *= feather_ratio
        mask[:, :, :, height - 1 - t: height - t] *= feather_ratio
        pass

    pass


def mask_step_alternative(feather: int, mask: torch.Tensor) -> None:
    width, height = mask.shape[2], mask.shape[3]
    inverted_feather = (1.0 / feather)

    for t in range(feather):
        feather_ratio = inverted_feather * (t + 1)

        # Erstellen Sie Maskenmatrizen für die betroffenen Bereiche
        mask_matrix_width = torch.ones_like(mask)
        mask_matrix_width[:, :, torch.tensor([t, width - 1 - t]), :] = feather_ratio

        mask_matrix_height = torch.ones_like(mask)
        mask_matrix_height[:, :, :, torch.tensor([t, height - 1 - t])] = feather_ratio

        # Wenden Sie die Masken auf den Tensor an
        mask *= mask_matrix_width
        mask *= mask_matrix_height

    pass


def mask_step_mixed(feather: int, mask: torch.Tensor) -> None:
    width, height = mask.shape[2], mask.shape[3]
    inverted_feather = (1.0 / feather)
    t = torch.arange(feather).float() + 1
    feather_ratios = inverted_feather * t

    # Erstellen Sie Maskenmatrizen für die betroffenen Bereiche
    mask_matrix_width = torch.ones_like(mask)
    mask_matrix_width[:, :, :feather, :] *= feather_ratios.view(1, 1, -1, 1)
    mask_matrix_width[:, :, width-feather:, :] *= feather_ratios.flip(0).view(1, 1, -1, 1)

    mask_matrix_height = torch.ones_like(mask)
    mask_matrix_height[:, :, :, :feather] *= feather_ratios.view(1, 1, 1, -1)
    mask_matrix_height[:, :, :, height-feather:] *= feather_ratios.flip(0).view(1, 1, 1, -1)

    # Wenden Sie die Masken auf den Tensor an
    mask *= mask_matrix_width
    mask *= mask_matrix_height
    pass
