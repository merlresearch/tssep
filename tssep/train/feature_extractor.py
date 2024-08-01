# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses

import torch.nn
from padertorch.contrib.cb.feature_extractor import *

from tssep.train.feature_extractor_torchaudio import TorchMFCC  # noqa


def interchannel_phase_differences(
    signal, second_channel=None, concatenate=False
):
    """
    Calculates the inter-channel phase differences:

        cos(angle(channel1 * channel2.conj()))
        sin(angle(channel1 * channel2.conj()))

    Args:
        signal: The stft signal.
            Shape: (..., channels, frames, features)
        second_channel:
            The corresponding second channel. When not given, use randomly
            sampled channels.
        concatenate:


    Returns:
            If concatenate True, return the concatenation of abs, cos and sin
            on the last axis.
            Otherwise, return the tuple (cos, sin)


    >>> np.random.seed(0)
    >>> signal = np.ones([6, 4, 5]) * np.exp(1j * np.random.uniform(0, 2*np.pi, [6, 1, 1])) * (np.arange(6)[:, None, None] + 1)
    >>> c, s = interchannel_phase_differences(signal)
    >>> c[0, :, :]
    array([[0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208]])
    >>> c[:, 0, 0], s[:, 0, 0]
    (array([0.81966208, 0.76070789, 0.93459697, 0.93459697, 0.72366352,
           0.90670355]), array([-0.57284734,  0.64909438,  0.35570844, -0.35570844, -0.69015296,
           -0.42176851]))
    >>> sig = interchannel_phase_differences(signal, concatenate=True)
    >>> sig[-1, 0, :]
    array([6.        , 6.        , 6.        , 6.        , 6.        ,
           0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208,
           0.57284734, 0.57284734, 0.57284734, 0.57284734, 0.57284734])

    >>> sig[:, 0, 0]
    array([1., 2., 3., 4., 5., 6.])

    """
    import itertools

    if second_channel is None:
        try:
            D = signal.shape[-3]
        except IndexError:
            raise IndexError(signal.shape)
        assert D >= 2, (D, signal.shape)
        l = list(itertools.permutations(range(D), 2))
        np.random.shuffle(l)
        second_channel = np.array(sorted(dict(l).items()))[:, 1]

    sincos = interchannel_phase_differences_op(
        signal, signal[..., second_channel, :, :]
    )

    if concatenate:
        return np.concatenate(
            [np.abs(signal), sincos.real, sincos.imag], axis=-1
        )
    else:
        return sincos.real, sincos.imag


class Log1pAbsIPDSTFT(AbsIPDSTFT):
    """
    >>> fe = Log1pAbsIPDSTFT()
    >>> fe
    Log1pAbsIPDSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=1539, window='blackman')
    >>> ref_channel = 0
    >>> np.squeeze(fe.stft_to_feature(np.array([[1, 5], [3+4j, -5]])[:, None, :]), axis=(-2))[ref_channel]
    array([ 0.69314718,  1.79175947,  0.6       , -1.        , -0.8       ,
            0.        ])
    >>> fe(np.ones((2, 10_000))).shape
    (2, 43, 1539)
    """

    def stft_to_feature(self, stft_signals):
        #  (channels, ..., frequencies)

        log1p = torch.log1p if torch.is_tensor(stft_signals) else np.log1p

        return np.concatenate(
            [
                log1p(abs(stft_signals)),
                *interchannel_phase_differences(
                    stft_signals, concatenate=False
                ),
            ],
            axis=-1,
        )


class MVNLog1pAbsSTFT(Log1pAbsSTFT):
    """
    >>> fe = MVNLog1pAbsSTFT()
    >>> fe
    MVNLog1pAbsSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513, window='blackman', norm_means=True, norm_vars=False, eps=1e-20)
    >>> Log1pAbsSTFT.stft_to_feature(fe, np.array([[1, 5], [3+4j, -5]]))
    array([[0.69314718, 1.79175947],
           [1.79175947, 1.79175947]])
    >>> fe.stft_to_feature(np.array([[1, 5], [3+4j, -5]]))
    array([[-0.54930614,  0.        ],
           [ 0.54930614,  0.        ]])
    >>> fe(np.ones(10_000)).shape
    (43, 513)
    """

    def __init__(
        self,
        size=1024,
        shift=256,
        window_length=None,
        pad=True,
        fading=True,
        output_size=None,
        window="blackman",
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20,
    ):
        super().__init__(
            size=size,
            shift=shift,
            window_length=window_length,
            pad=pad,
            fading=fading,
            output_size=output_size,
            window=window,
        )

        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def stft_to_feature(self, stft_signals):
        feature = super().stft_to_feature(stft_signals)

        # feature shape: ..., frame, freq

        # Do the same as https://github.com/espnet/espnet/blob/master/espnet2/layers/utterance_mvn.py
        if self.norm_means:
            mean = np.mean(feature, axis=-2, keepdims=True)
            feature -= mean
            if self.norm_vars:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        return feature


class NoFeatureSTFT(STFT):
    def stft_to_feature(self, stft_signals):
        return stft_signals[..., :0]

    def _get_output_size(self, output_size):
        if output_size is None:
            return 0
        else:
            assert output_size == 0, (output_size, self.frequencies)
            return output_size


class Log1pMaxNormAbsSTFT(STFT):
    """
    Math:
        f(y) = log(1 + abs(STFT(y)) / c) in [0, 1]
        where
            c = (np.e - 1) / max(abs(STFT(y), axis=statistics_axis)


    >>> fe = Log1pMaxNormAbsSTFT()
    >>> fe
    Log1pMaxNormAbsSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513, window='blackman', statistics_axis='tf')
    >>> fe.stft_to_feature(np.array([[1, 5], [3+4j, -5]]))
    array([[0.29539453, 1.        ],
           [1.        , 1.        ]])
    >>> rng = np.random.RandomState(0)
    >>> f = fe(rng.uniform(0, 1, size=10_000))
    >>> f.shape
    (43, 513)
    >>> np.mean(f), np.min(f), np.max(f), np.std(f)
    (0.03461471931132962, 1.0003006801514706e-06, 1.0, 0.051645387514742555)

    >>> np.log1p([0, np.e-1])
    array([0., 1.])
    """

    sample_index_to_frame_index = pb.transform.STFT.sample_index_to_frame_index

    def __init__(
        self,
        size=1024,
        shift=256,
        window_length=None,
        pad=True,
        fading=True,
        output_size=None,
        window="blackman",
        statistics_axis="tf",
    ):
        super().__init__(
            size=size,
            shift=shift,
            window_length=window_length,
            pad=pad,
            fading=fading,
            output_size=output_size,
            window=window,
        )

        self.statistics_axis = statistics_axis

    def stft_to_feature(self, stft_signals):
        if pt_complex.is_torch(stft_signals):
            s = stft_signals
            s = abs(s)
            if self.statistics_axis == "tf":
                norm = torch.amax(s, keepdim=True, dim=(-2, -1))
            elif self.statistics_axis == "t":
                norm = torch.amax(s, keepdim=True, dim=-2)
            elif self.statistics_axis == "f":
                norm = torch.amax(s, keepdim=True, dim=-1)
            else:
                raise ValueError(self.statistics_axis)

            s *= (np.e - 1) / norm

            return torch.log1p(s)
        else:
            s = stft_signals
            s = abs(s)
            if self.statistics_axis == "tf":
                norm = np.amax(s, keepdims=True, axis=(-2, -1))
            elif self.statistics_axis == "t":
                norm = np.amax(s, keepdims=True, axis=-2)
            elif self.statistics_axis == "f":
                norm = np.amax(s, keepdims=True, axis=-1)
            else:
                raise ValueError(self.statistics_axis)

            s *= (np.e - 1) / norm

            return np.log1p(s)


class Log1pMaxNormAbsIPDSTFT(Log1pMaxNormAbsSTFT):
    def _get_output_size(self, output_size):
        if output_size is None:
            return (self.size // 2 + 1) * 3
        else:
            assert output_size == self.frequencies * 3, (
                output_size,
                self.frequencies * 3,
            )
            return output_size

    def stft_to_feature(self, stft_signals):
        signal = super().stft_to_feature(stft_signals)
        return np.concatenate(
            [
                signal,
                *interchannel_phase_differences(
                    stft_signals, concatenate=False
                ),
            ],
            axis=-1,
        )


class ConcaternatedSTFTFeatures(STFT, torch.nn.Module):
    """

    >>> fe1 = Log1pMaxNormAbsSTFT()
    >>> fe1
    Log1pMaxNormAbsSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513, window='blackman', statistics_axis='tf')
    >>> fe1.stft_to_feature(np.array([[1, 5], [3+4j, -5]]))
    array([[0.29539453, 1.        ],
           [1.        , 1.        ]])
    >>> fe = ConcaternatedSTFTFeatures(fe1, Log1pAbsSTFT())
    >>> fe.stft_to_feature(np.array([[1, 5], [3+4j, -5]]))
    array([[0.29539453, 1.        , 0.69314718, 1.79175947],
           [1.        , 1.        , 1.79175947, 1.79175947]])

    """

    sample_index_to_frame_index = pb.transform.STFT.sample_index_to_frame_index

    @classmethod
    def finalize_dogmatic_config(cls, config):
        for fe in ["fe1", "fe2"]:
            config[fe]["size"] = config["size"]
            config[fe]["shift"] = config["shift"]
            config[fe]["pad"] = config["pad"]
            config[fe]["fading"] = config["fading"]
            config[fe]["window"] = config["window"]
            if config["window_length"] is not None:
                config[fe]["window_length"] = config["window_length"]

        super().finalize_dogmatic_config(config)
        for fe in ["fe1", "fe2"]:
            config[fe]["window_length"] = config["window_length"]

    def __init__(
        self,
        fe1,
        fe2,
        output_size=None,
        size=1024,
        shift=256,
        window="blackman",
        window_length=None,
        pad=True,
        fading=True,
    ):
        # Torch complains when the assignment is done here, but super calls _get_output_size
        # and _get_output_size needs fe1 and fe2
        # AttributeError: cannot assign module before Module.__init__() call
        self._tmp = [fe1, fe2]

        super().__init__(
            size=size,
            shift=shift,
            window_length=window_length,
            pad=pad,
            fading=fading,
            output_size=output_size,
            window=window,
        )
        self.fe1 = fe1
        self.fe2 = fe2

    def stft_to_feature(self, stft_signals):
        feature = [
            self.fe1.stft_to_feature(stft_signals),
            self.fe2.stft_to_feature(stft_signals),
        ]
        if isinstance(stft_signals, np.ndarray):
            return np.concatenate(feature, axis=-1)
        else:
            return torch.concat(feature, dim=-1)

    def _get_output_size(self, output_size):
        fe1, fe2 = self._tmp
        if output_size is None:
            return fe1._get_output_size(None) + fe2._get_output_size(None)
        else:
            return output_size


class KaldiTorch(STFT, torch.nn.Module):
    def __init__(
        self,
        func,
        fe: STFT,
    ):
        super().__init__()
        self.func = func
        self.fe = fe

    def _get_output_size(self, output_size):
        return self.fe._get_output_size(output_size)

    def stft(self, signal):
        return self.feature_extractor.stft(signal)

    def istft(self, signal, num_samples=None):
        return self.feature_extractor.istft(signal, num_samples=num_samples)

    def stft_to_feature(self, stft_signals):
        return self.func(self.fe.stft_to_feature(stft_signals))


@dataclasses.dataclass
class KaldiTorchMFCC(AbstractFeatureExtractor):
    blackman_coeff: float = 0.42
    cepstral_lifter: float = 22.0
    channel: int = -1
    dither: float = 0.0
    energy_floor: float = 1.0
    frame_length: float = 25.0
    frame_shift: float = 10.0
    high_freq: float = 0.0
    htk_compat: bool = False
    low_freq: float = 20.0
    num_ceps: int = 13
    min_duration: float = 0.0
    num_mel_bins: int = 23
    preemphasis_coefficient: float = 0.97
    raw_energy: bool = True
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    sample_frequency: float = 16000.0
    snip_edges: bool = True
    subtract_mean: bool = False
    use_energy: bool = False
    vtln_high: float = -500.0
    vtln_low: float = 100.0
    vtln_warp: float = 1.0
    window_type: str = "povey"

    def __call__(self, signals):
        import torchaudio

        return torchaudio.compliance.kaldi.mfcc(
            signals,
            **dataclasses.asdict(self),
        )
