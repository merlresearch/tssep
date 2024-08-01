# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import typing

import numpy as np
import paderbox as pb
import padertorch as pt
import pb_bss
import pb_bss.testing.random_utils
import torch
from einops import rearrange

import tssep.train.enhancer_distortion_mask

if typing.TYPE_CHECKING:
    import tssep.train.model


class ABC(pt.Configurable, torch.nn.Module):
    @property
    def name(self):
        return self.__class__.__name__

    def __call__(
        self,
        masks: torch.Tensor,  # spk mask time freq
        ex,
        model: "tssep.train.model.Model",
    ):
        raise NotImplementedError()


class Dummy(ABC):
    def __call__(
        self,
        masks: torch.Tensor,  # spk mask time freq
        ex,
        model: "tssep.train.model.Model",
    ):
        return None


class Nothing(ABC):
    def __call__(
        self,
        masks: torch.Tensor,  # spk mask time freq
        ex,
        model: "tssep.train.model.Model",
    ):
        reference_channel = ex["reference_channel"]
        Observation = ex["Observation"]

        batched = {4: False, 5: True}[len(masks.shape)]

        if reference_channel is None:
            assert len(Observation.shape) == (
                3 if batched else 2
            ), Observation.shape
        else:
            assert len(Observation.shape) == (
                4 if batched else 3
            ), Observation.shape
            Observation = Observation[..., reference_channel, :, :]

        if isinstance(Observation, np.ndarray):
            Observation = torch.tensor(Observation, device=masks.device)

        return Observation[..., None, :, :]  # [..., 0, :, :]


class Masking(ABC):
    def __call__(
        self,
        masks: torch.Tensor,  # spk mask time freq
        ex,
        model: "tssep.train.model.Model",
    ):
        reference_channel = ex["reference_channel"]
        Observation = ex["Observation"]

        batched = {4: False, 5: True}[len(masks.shape)]

        if reference_channel is None:
            assert len(Observation.shape) == (
                3 if batched else 2
            ), Observation.shape
        else:
            assert len(Observation.shape) == (
                4 if batched else 3
            ), Observation.shape
            Observation = Observation[..., reference_channel, :, :]

        if isinstance(Observation, np.ndarray):
            Observation = torch.tensor(Observation, device=masks.device)

        return Observation[..., None, :, :] * torch.squeeze(
            masks, dim=-3
        )  # [..., 0, :, :]


def trace(input, axis1=-2, axis2=-1):
    """
    https://github.com/pytorch/pytorch/issues/52668

    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    >>> torch.trace(x)
    tensor(15.)
    >>> torch.trace(x.view(1, 3, 3))
    Traceback (most recent call last):
    ...
    RuntimeError: trace: expected a matrix, but got tensor with dim 3
    >>> trace(x)
    tensor(15.)
    >>> trace(x.view(3, 3, 1), axis1=0, axis2=1)
    tensor([15.])
    >>> trace(x.view(1, 3, 3), axis1=2, axis2=1)
    tensor([15.])
    >>> trace(x.view(3, 1, 3), axis1=0, axis2=2)
    tensor([15.])
    """
    assert input.shape[axis1] == input.shape[axis2], input.shape

    shape = list(input.shape)
    strides = list(input.stride())
    strides[axis1] += strides[axis2]

    shape[axis2] = 1
    strides[axis2] = 0

    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(axis1, axis2))


class TorchBF(ABC):
    def __init__(
        self,
        bf="mvdr_souden",
        masking=False,
        masking_eps=0.0,
        eps=None,
    ):
        super().__init__()
        assert bf == "mvdr_souden", (bf, "Only mvdr_souden is implemented")
        self.bf = bf
        self.eps = eps
        self.masking = masking
        self.masking_eps = masking_eps

    def __call__(
        self,
        masks,  # spk mask time freq
        ex,
        model: "tssep.train.model.Model",
    ):
        """
        >>> import paderbox as pb
        >>> from tssep.train.feature_extractor import STFT
        >>> from ci_sdr.pt.sdr import ci_sdr
        >>> stft = STFT(size=1024, shift=256)
        >>> from paderbox.testing.testfile_fetcher import get_file_path
        >>> y = pb.io.load(get_file_path('observation.wav'))
        >>> s0 = pb.io.load(get_file_path('speech_source_0.wav'))
        >>> s1 = pb.io.load(get_file_path('speech_source_1.wav'))
        >>> x0 = pb.io.load(get_file_path('speech_image_0.wav'))
        >>> x1 = pb.io.load(get_file_path('speech_image_1.wav'))
        >>> X = stft(np.array([x0, x1]))
        >>> N = stft(pb.io.load(get_file_path('noise_image.wav')))
        >>> from pb_bss.extraction.mask_module import wiener_like_mask
        >>> from pb_bss.evaluation import OutputMetrics, InputMetrics
        >>> M0, M1, MN = wiener_like_mask([*X, N], sensor_axis=1)
        >>> M0.shape
        (154, 513)
        >>> bf = TorchBF('mvdr_souden')
        >>> ex = {'Observation': torch.tensor(stft(y)), 'reference_channel': 0}
        >>> pb.utils.pretty.pprint(ex)
        {'Observation': tensor(shape=(6, 154, 513), dtype=complex128),
         'reference_channel': 0}
        >>> pb.utils.pretty.pprint(torch.tensor(np.array([[M0, M1+MN], [M1, M0+MN]]), dtype=torch.float32))
        tensor(shape=(2, 2, 154, 513), dtype=float32)

        # Call with target and inference mask
        >>> est = bf(torch.tensor(np.array([[M0, M1+MN], [M1, M0+MN]]), dtype=torch.float32), ex, None)
        >>> est.shape
        torch.Size([2, 154, 513])
        >>> ci_sdr(torch.tensor([s0, s1]), torch.tensor(stft.istft(est, num_samples=y.shape[-1])), compute_permutation=False)
        tensor([23.6640, 20.0154], dtype=torch.float64)

        # Call with ignoring cross talker in inference mask -> Performance is worse
        >>> est = bf(torch.tensor(np.array([[M0, MN], [M1, MN]])), ex, None)
        >>> est.shape
        torch.Size([2, 154, 513])
        >>> ci_sdr(torch.tensor([s0, s1]), torch.tensor(stft.istft(est, num_samples=y.shape[-1])), compute_permutation=False)
        tensor([17.4672,  9.7333], dtype=torch.float64)

        # Call with target mask only (inference mask is set to 1-target mask)
        >>> est = bf(torch.tensor(np.array([[M0], [M1]])), ex, None)
        >>> est.shape
        torch.Size([2, 154, 513])
        >>> ci_sdr(torch.tensor([s0, s1]), torch.tensor(stft.istft(est, num_samples=y.shape[-1])), compute_permutation=False)
        tensor([23.6640, 20.0154], dtype=torch.float64)

        >>> bf = TorchBF('mvdr_souden', masking=True, masking_eps=0.1)
        >>> est = bf(torch.tensor(np.array([[M0], [M1]])), ex, None)
        >>> est.shape
        torch.Size([2, 154, 513])
        >>> ci_sdr(torch.tensor([s0, s1]), torch.tensor(stft.istft(est, num_samples=y.shape[-1])), compute_permutation=False)
        tensor([18.6837, 15.4242], dtype=torch.float64)

        """
        batched = {4: False, 5: True}[len(masks.shape)]
        reference_channel = ex["reference_channel"]
        Observation = ex["Observation"]

        assert len(Observation.shape) == (
            4 if batched else 3
        ), Observation.shape
        assert Observation.dtype == torch.complex128, Observation.dtype

        if masks.shape[-3] == 2:
            psds = torch.einsum(
                "...kmtf,...dtf,...Dtf->...mkfdD",
                masks.to(torch.complex128),
                Observation,
                Observation.conj(),
            )
            target_psd = psds[..., 0, :, :, :, :]
            interference_psd = psds[..., 1, :, :, :, :]
        elif masks.shape[-3] == 1:
            m = torch.squeeze(masks, dim=-3).to(torch.complex128)
            target_psd = torch.einsum(
                "...ktf,...dtf,...Dtf->...kfdD",
                m,
                Observation,
                Observation.conj(),
            )
            interference_psd = torch.einsum(
                "...ktf,...dtf,...Dtf->...kfdD",
                1 - m,
                Observation,
                Observation.conj(),
            )
        else:
            raise ValueError(masks.shape)
        phi = torch.linalg.solve(interference_psd, target_psd)
        lambda_ = trace(phi)[..., None, None]

        eps = torch.finfo(lambda_.dtype).tiny if self.eps is None else self.eps
        mat = phi / torch.clamp(lambda_.real, min=eps)
        beamformer = mat[..., reference_channel]
        enh = torch.einsum(
            "...kfd,...dtf->...ktf", beamformer.conj(), Observation
        )

        if self.masking:
            enh = enh * torch.clamp(
                masks[..., :, 0, :, :], min=self.masking_eps
            )

        return enh


def _get_psd(mask, observation, mask_power):
    # Assuming a different shape as in 'pb_bss.extraction.beamformer.get_power_spectral_density_matrix'
    # simplifies this function and makes it easier to use broadcasting.

    # t ... time frame
    # f ... frequency
    # d ... microphone

    if mask_power != 1:
        mask = mask**mask_power
    else:
        assert mask_power > 0, mask_power

    psd = (
        np.einsum(
            "...t,...dt,...Dt->...dD", mask, observation, observation.conj()
        )
        / observation.shape[-1]
    )

    psd = (psd + np.swapaxes(psd, -2, -1)) / 2
    return psd


class WPE:
    def __init__(
        self,
        taps=10,
        delay=2,
        iterations=3,
        psd_context=0,
        statistics_mode="full",
    ):
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.psd_context = psd_context
        self.statistics_mode = statistics_mode

    def __call__(self, Observation, inplace=False):
        """

        >>> Y = np.random.normal(size=(3, 40, 5))
        >>> Z_np = WPE()(Y)
        >>> Z_pt = WPE()(torch.tensor(Y))
        >>> np.testing.assert_allclose(Z_np, Z_pt, rtol=1e-6, atol=1e-6)

        """
        if isinstance(Observation, np.ndarray):
            import nara_wpe.wpe

            return rearrange(
                nara_wpe.wpe.wpe_v8(
                    Y=rearrange(Observation, "d t f -> f d t"),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                    statistics_mode=self.statistics_mode,
                    inplace=inplace,
                ),
                "f d t -> d t f",
            )

        if isinstance(Observation, torch.Tensor):
            import nara_wpe.torch_wpe

            return rearrange(
                nara_wpe.torch_wpe.wpe_v6(
                    Y=rearrange(Observation, "d t f -> f d t"),
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                    statistics_mode=self.statistics_mode,
                    # inplace=inplace,  # not implemented
                ),
                "f d t -> d t f",
            )

        raise NotImplementedError(type(Observation), Observation)


class ChannelWiseWPE(WPE):
    def __call__(self, Observation, inplace=False):
        """

        >>> Y = np.random.normal(size=(3, 40, 5))
        >>> Z_np = torch.squeeze(torch.stack([WPE()(Y_[None, :, :]) for Y_ in torch.tensor(Y)]), dim=1)
        >>> Z_pt = ChannelWiseWPE()(torch.tensor(Y))
        >>> np.testing.assert_allclose(Z_np, Z_pt, rtol=1e-6, atol=1e-6)

        """
        return rearrange(
            super().__call__(
                rearrange(Observation, "d t f -> 1 t (d f)"), inplace=inplace
            ),
            "1 t (d f) -> d t f",
            d=Observation.shape[0],
        )


class ClassicBF_np(ABC):
    """
    Eval enhancer

    >>> from paderbox.utils.pretty import pprint
    >>> from paderbox.array.interval import ArrayInterval
    >>> from tssep.data import simple_toy_example
    >>> ex = simple_toy_example(frequency_bins=17)
    >>> pprint(ex)
    {'Observation': array(shape=(6, 79, 17), dtype=complex128),
     'Speech_reverberation_early': array(shape=(2, 6, 79, 17), dtype=complex128),
     'Vad': [ArrayInterval("0:55", shape=(79,)),
      ArrayInterval("45:79", shape=(79,))],
     'mask': array(shape=(3, 79, 17), dtype=float64)}

    >>> pprint(ClassicBF_np.get_config())
    {'factory': 'tssep.train.enhancer.ClassicBF_np',
     'bf': 'mvdr_souden',
     'masking': False,
     'masking_eps': 0,
     'distortion_mask': {'factory': 'tssep.train.enhancer_distortion_mask.SumCrossTalker',
      'eps': 0.0001},
     'pre_wpe': None,
     'segment_wpe': None,
     'mask_power': 1}
    >>> enh = ClassicBF_np.new()
    >>> enh
    ClassicBF_np()
    >>> estimate = enh(ex['mask'][:-1, None, :, :], ex['Observation'], ex['Vad'], numpy_out=True)
    >>> pprint(estimate)
    array(shape=(2, 79, 17), dtype=complex128)
    >>> list(map(ArrayInterval, abs(estimate).sum(axis=-1) != 0))
    [ArrayInterval("0:55", shape=(79,)), ArrayInterval("45:79", shape=(79,))]

    >>> from tssep.train.feature_extractor import STFT
    >>> from pb_bss.evaluation import OutputMetrics, InputMetrics
    >>> stft = STFT(size=32, shift=32, window='boxcar')

    >>> kwargs = dict(
    ...     # Speech_reverberation_early is the wrong target for evaluation, but sufficient for this test
    ...     speech_source=stft.istft(ex['Speech_reverberation_early'][:, 0]),
    ...     sample_rate=16000,
    ... )

    >>> pprint(InputMetrics(stft.istft(ex['Observation']), **kwargs).mir_eval_sdr)
    array([[ 3.17085822,  2.32028164,  2.65168557,  2.69398294,  2.68378942,
             2.51380272],
           [-1.76406267, -1.95131548, -4.66883486, -3.48070751, -3.38314618,
            -2.20398073]])
    >>> pprint(OutputMetrics(stft.istft(estimate), **kwargs, compute_permutation=False).mir_eval_sdr)
    array([8.46594551, 8.59388547])
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        # To activate WPE, set the factory of pre_wpe or segment_wpe to WPE
        # config['pre_wpe'] = {'factory': WPE}
        # config['segment_wpe'] = {'factory': WPE}
        config["distortion_mask"] = {
            "factory": tssep.train.enhancer_distortion_mask.SumCrossTalker
        }

    def __init__(
        self,
        bf="mvdr_souden",
        masking=False,
        masking_eps=0,
        distortion_mask: "tssep.train.enhancer_distortion_mask.SumCrossTalker" = None,
        pre_wpe: WPE = None,
        segment_wpe: WPE = None,
        mask_power: "int | float" = 1,
    ):
        super().__init__()
        self.bf = bf
        self.masking = masking
        self.masking_eps = masking_eps
        self.distortion_mask = distortion_mask
        self.mask_power = mask_power
        self.pre_wpe = pre_wpe
        self.segment_wpe = segment_wpe

    def __call__(
        self,
        masks: torch.Tensor,  # spk mask time freq
        Observation,
        dia,
        segment_bf=True,
        numpy_out=False,
    ):
        masks = pt.utils.to_numpy(masks)
        Observation = pt.utils.to_numpy(Observation)
        mics = Observation.shape[0]
        assert mics >= 6 or self.bf in [
            "ch0",
            "ch1",
        ], (
            Observation.shape
        )  # Check that all channels are loaded. At training time usually only one or two are used

        if self.pre_wpe:
            Observation = self.pre_wpe(Observation)

        Observation = rearrange(Observation, "mic time freq -> freq mic time")
        masks = rearrange(masks, "spk mask time freq -> mask spk freq time")

        _, K, F, T = masks.shape

        if masks.shape[0] == 1 or self.bf == "ch0":
            if self.bf == "ch0":
                masks = masks[:1]
            masks = self.distortion_mask(masks)
        else:
            assert masks.shape[0] == 2, masks.shape
            raise NotImplementedError(masks.shape)

        if dia is None:
            assert segment_bf is False, segment_bf
            assert self.segment_wpe is None, self.segment_wpe
            assert numpy_out is True, numpy_out
            dia = [None] * K

        assert isinstance(dia, (tuple, list)), (
            "Expect list of ArrayInterval",
            type(dia),
            dia,
        )

        bf_kwargs = pb.utils.mapping.Dispatcher(
            {
                "mvdr_souden": dict(ref_channel=0),
                "scaled_gev_atf+mvdr": dict(ref_channel=0),
                "rank1_gev+mvdr_souden": dict(ref_channel=0),
                "wmwf": dict(reference_channel=0),
                "ch0": dict(),
                "ch1": dict(),
            }
        )[self.bf]

        ret = []
        if numpy_out:
            out = np.zeros([K, T, F], dtype=Observation.dtype)
        for target_idx, ai in enumerate(dia):
            ret_spk = {}

            if segment_bf:
                assert isinstance(ai, pb.array.interval.ArrayInterval), (
                    type(ai),
                    ai,
                )
                for s, e in ai.normalized_intervals:

                    Observation_local = Observation[:, :, s:e]

                    if self.segment_wpe:
                        Observation_local = self.segment_wpe(Observation_local)

                    psd_target, psd_distortion = _get_psd(
                        masks[:, target_idx, :, s:e],
                        Observation_local,
                        mask_power=self.mask_power,
                    )
                    bf = pb_bss.extraction.get_bf_vector(
                        self.bf, psd_target, psd_distortion, **bf_kwargs
                    )

                    ret_spk[(s, e)] = rearrange(
                        pb_bss.extraction.apply_beamforming_vector(
                            bf, Observation_local
                        ),
                        "freq time -> time freq",
                    )

                    if self.masking:
                        ret_spk[(s, e)] = ret_spk[(s, e)] * np.maximum(
                            masks[0, target_idx, :, s:e].T, self.masking_eps
                        )

                    if numpy_out:
                        out[target_idx, s:e, :] = ret_spk[(s, e)]
                ret.append(ret_spk)
            else:
                assert self.segment_wpe is None, (
                    self.segment_wpe,
                    self.segment_bf,
                )

                psd_target, psd_distortion = _get_psd(
                    masks[:, target_idx, :, :],
                    Observation[:, :, :],
                    mask_power=self.mask_power,
                )
                bf = pb_bss.extraction.get_bf_vector(
                    self.bf, psd_target, psd_distortion, **bf_kwargs
                )

                if ai is None:
                    assert numpy_out is True, numpy_out

                    out[target_idx, :, :] = rearrange(
                        pb_bss.extraction.apply_beamforming_vector(
                            bf, Observation[:, :, :]
                        ),
                        "freq time -> time freq",
                    )

                else:
                    for s, e in ai.normalized_intervals:
                        ret_spk[(s, e)] = rearrange(
                            pb_bss.extraction.apply_beamforming_vector(
                                bf, Observation[:, :, s:e]
                            ),
                            "freq time -> time freq",
                        )
                        if numpy_out:
                            out[target_idx, s:e, :] = ret_spk[(s, e)]
                    raise NotImplementedError("ToDo")

        if numpy_out:
            return out
        else:
            return ret
