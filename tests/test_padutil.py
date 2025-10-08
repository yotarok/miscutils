import torch

from miscutils import padutil


def test_pack_and_pad_right():
    x = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [2, 4, 6, 8, 10, 12, 14],
            [3, 6, 9, 12, 15, 18, 21],
            [4, 8, 12, 16, 20, 24, 28],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, True, False, True, False, True],
            [False, True, True, True, False, True, False],
            [True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    y, ymask = padutil.pack_and_pad_right(x, mask)

    y_list = [
        [v.item() for v, m in zip(y_row, ymask_row) if m]
        for y_row, ymask_row in zip(y, ymask)
    ]

    assert len(y_list) == 4
    assert y_list[0] == [1, 3, 5, 7]
    assert y_list[1] == [4, 6, 8, 12]
    assert y_list[2] == [3, 6, 9]
    assert y_list[3] == [4, 8, 12, 16, 20, 24, 28]
    padutil.right_pad_mask_to_length(ymask, verify=True)


def test_mask_last_n_elems():
    mask = torch.tensor(
        [
            [True, False, True, False, True, False, True],
            [False, True, True, True, False, True, False],
            [True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )

    nmask = padutil.mask_last_n_elems(1, mask).tolist()

    assert nmask[0] == [True, False, True, False, True, False, False]
    assert nmask[1] == [False, True, True, True, False, False, False]
    assert nmask[2] == [True, True, False, False, False, False, False]
    assert nmask[3] == [True, True, True, True, True, True, False]
