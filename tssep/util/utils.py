# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np
import paderbox as pb
import torch


def stft_vad(
    vad,
    window_length,
    shift,
    fading,
):
    """
    Move a sample activity to frame activity
    """
    from paderbox.transform.module_stft import (
        _samples_to_stft_frames,
        sample_index_to_stft_frame_index,
    )

    if isinstance(vad, torch.Tensor):
        return torch.Tensor(
            stft_vad(vad.cpu().numpy(), window_length, shift, fading)
        ).to(vad.device)

    if isinstance(vad, (np.ndarray, pb.array.interval.ArrayInterval)):
        data = np.empty(vad.shape[:-1], dtype=object)
        for idx in np.ndindex(vad.shape[:-1]):

            a = vad[idx]
            ai = pb.array.interval.zeros(
                _samples_to_stft_frames(
                    len(a),
                    size=window_length,
                    shift=shift,
                    pad=True,
                    fading=fading,
                )
            )

            normalized_intervals = pb.array.interval.ArrayInterval(
                a
            ).normalized_intervals
            if (
                normalized_intervals
            ):  # normalized_intervals is empty when a has no True value
                starts, ends = np.array(normalized_intervals).T

                starts = sample_index_to_stft_frame_index(
                    starts,
                    window_length=window_length,
                    shift=shift,
                    fading=fading,
                )
                ends = sample_index_to_stft_frame_index(
                    ends,
                    window_length=window_length,
                    shift=shift,
                    fading=fading,
                )

                for start, end in zip(starts, ends):
                    ai[start:end] = True

            data[idx] = ai
        if isinstance(vad, pb.array.interval.ArrayInterval):
            return data.tolist()
        else:
            return np.array(data.tolist())
    elif isinstance(vad, (tuple, list)):
        return [stft_vad(v, window_length, shift, fading) for v in vad]
    else:
        raise TypeError(vad)


def istft_vad(
    vad,
    window_length,
    shift,
    fading,
):
    """
    Move a frame activity to sample activity
    """
    from paderbox.transform.module_stft import stft_frame_index_to_sample_index

    if isinstance(vad, (np.ndarray, pb.array.interval.ArrayInterval)):
        data = np.empty(vad.shape[:-1], dtype=object)
        for idx in np.ndindex(vad.shape[:-1]):

            a = vad[idx]
            ai = pb.array.interval.zeros()

            normalized_intervals = pb.array.interval.ArrayInterval(
                a
            ).normalized_intervals
            if (
                normalized_intervals
            ):  # normalized_intervals is empty when a has no True value
                starts, ends = np.array(normalized_intervals).T

                starts = stft_frame_index_to_sample_index(
                    starts,
                    window_length=window_length,
                    shift=shift,
                    fading=fading,
                    mode="first",
                )
                ends = stft_frame_index_to_sample_index(
                    ends,
                    window_length=window_length,
                    shift=shift,
                    fading=fading,
                    mode="last",
                )

                for start, end in zip(starts, ends):
                    ai[start:end] = True

            data[idx] = ai
        return data.tolist()
    elif isinstance(vad, (tuple, list)):
        return [stft_vad(v, window_length, shift, fading) for v in vad]
    else:
        raise TypeError(vad)
