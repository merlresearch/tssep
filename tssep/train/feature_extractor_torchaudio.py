# Copyright (c) 2024 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2022 torchaudio developers (BSD 2-Clause License)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-2-Clause
import paderbox as pb
import torch
from padertorch.contrib.cb.feature_extractor import STFT


class TorchMFCC(STFT, torch.nn.Module):
    """
    This class calculates the MFCC like torchaudio.transforms.MFCC,
    but with the following differences:
     - It allows to have the STFT as input.
     - It uses pb.transform.STFT and not the STFT from torch.

    """

    sample_index_to_frame_index = pb.transform.STFT.sample_index_to_frame_index

    def __init__(
        self,
        size=400,
        shift=200,
        window_length=None,
        pad=True,
        fading=True,
        output_size=None,
        window="hann",
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_norm: str = "ortho",
        log_mels: bool = False,
        f_min: float = 40,
        f_max: float = -400,
        n_mels: int = 40,
        mel_norm: str = None,
        mel_scale: str = "htk",
    ):
        import torchaudio

        self.n_mfcc = n_mfcc

        super().__init__(
            size=size,
            shift=shift,
            window_length=window_length,
            pad=pad,
            fading=fading,
            output_size=output_size,
            window=window,
        )
        self.sample_rate = sample_rate
        self.f_min = f_min

        if f_max and f_max < 0:
            f_max = sample_rate + f_max

        self.f_max = f_max
        self.n_mels = n_mels
        self.dct_norm = dct_norm
        self.mel_norm = mel_norm
        self.mel_scale = mel_scale

        self.top_db = 80  # Also fixed in torchaudio
        self.log_mels = log_mels

        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB(
            "power", self.top_db
        )

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels,
            sample_rate,
            f_min,
            f_max,
            size // 2 + 1,
            mel_norm,
            mel_scale,
        )
        dct_mat = torchaudio.functional.create_dct(
            n_mfcc, self.mel_scale.n_mels, dct_norm
        )
        self.register_buffer("dct_mat", dct_mat)

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.n_mfcc
        else:
            return output_size

    def stft_to_feature(self, stft_signals):
        mel_specgram = self.mel_scale(
            abs(stft_signals.transpose(-1, -2)).to(torch.float32) ** 2
        )

        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., time, n_nfcc)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat)
        return mfcc
