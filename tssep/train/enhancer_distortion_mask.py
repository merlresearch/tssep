# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np


class OneMinus:
    """
    >>> m = np.array([0, 0.5, 1])[None]
    >>> OneMinus()(m)
    array([[0. , 0.5, 1. ],
           [1. , 0.5, 0. ]])
    """

    def __call__(self, masks):
        assert masks.shape[0] == 1, masks.shape
        noise_mask = np.maximum(1 - masks, 0)
        masks = np.concatenate([masks, noise_mask], axis=0)
        return masks


class SumCrossTalker:
    """
    >>> m = np.array([[0, 0.2, 0.8, 1, 0], [0.1, 0, 0.5, 1, 0], [1, 0.1, 1, 0.5, 0]])[None, :, :, None]
    >>> np.squeeze(SumCrossTalker(eps=0.01)(m))
    array([[[0.  , 0.2 , 0.8 , 1.  , 0.  ],
            [0.1 , 0.  , 0.5 , 1.  , 0.  ],
            [1.  , 0.1 , 1.  , 0.5 , 0.  ]],
    <BLANKLINE>
           [[1.1 , 0.1 , 1.5 , 1.5 , 0.01],
            [1.  , 0.3 , 1.8 , 1.5 , 0.01],
            [0.1 , 0.2 , 1.3 , 2.  , 0.01]]])
    """

    def __init__(self, eps=0.0001):
        self.eps = eps

    def __call__(self, masks):
        assert masks.shape[0] == 1, masks.shape
        # mask spk freq time

        speakers = masks.shape[1]
        noise_mask = np.stack(
            [
                np.sum(np.delete(masks, spk, axis=1), axis=1)
                for spk in range(speakers)
            ],
            axis=1,
        )

        noise_mask = np.maximum(noise_mask, self.eps)
        masks = np.concatenate([masks, noise_mask], axis=0)
        return masks
