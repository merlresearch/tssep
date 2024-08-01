# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import inspect

import einops
import numpy as np
import padertorch as pt
import torch.nn.functional
from padertorch.contrib.cb.summary import ReviewSummary

import tssep.train.feature_extractor
import tssep.train.model


class ABC(torch.nn.Module):
    def __init__(
        self,
        target: str = "speaker_reverberation_early_ch0",
        pit: bool = False,
    ):
        super().__init__()
        self.target = target
        self.pit = pit

    def _upper(self, s: str):
        return s[0].upper() + s[1:]

    def targets(self, lower=False, upper=False):
        if lower:
            assert not upper, (lower, upper)
            # Generic implementation so subclasses can call super
            return tuple([t.lower() for t in self.targets()])
        if upper:
            assert not lower, (lower, upper)
            # Generic implementation so subclasses can call super
            return tuple([self._upper(t) for t in self.targets()])
        else:
            return (self.target,)

    def extra_repr(self) -> str:
        sig = inspect.signature(self.__class__)
        return ", ".join(
            [
                f"{p.name}={getattr(self, p.name)!r}"
                for p in sig.parameters.values()
            ]
        )

    @property
    def name(self):
        return self.__class__.__name__

    def loss_fn(self, estimate, target):
        raise NotImplementedError()

    def forward(self, estimate, target):
        # Example shapes: [2, 236080], [2, 236080]
        assert estimate.shape == target.shape, (estimate.shape, target.shape)

        if self.pit:
            # This is the simple PIT implementation that works for every loss out of the box.
            return pt.ops.losses.pit_loss(
                estimate, target, axis=-2, loss_fn=self.loss_fn
            )
        else:
            return self.loss_fn(estimate, target)

    def from_ex_out(
        self,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
        summary: "pt.contrib.cb.summary.ReviewSummary",
    ):
        raise NotImplementedError

    def update_summary(
        self,
        summary: ReviewSummary,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
    ):
        pass


class TimeDomain(ABC):
    def from_ex_out(
        self,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
        summary: "pt.contrib.cb.summary.ReviewSummary",
    ):
        estimate = out.time_estimate
        target = ex[self.target]
        return self(estimate, target)


class STFTDomain(ABC):
    def from_ex_out(
        self,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
        summary: "pt.contrib.cb.summary.ReviewSummary",
    ):
        estimate = out.stft_estimate
        assert self.target[0].isupper(), self.target
        if self.target not in ex:
            ex[self.target] = self.fe.stft(ex[self.target.lower()])
        target = ex[self.target]
        return self(estimate, target)


class LogitsSTFTDomain(ABC):
    def prepare_target(self, target):
        raise NotImplementedError(self.__class__)

    def from_ex_out(
        self,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
        summary: "pt.contrib.cb.summary.ReviewSummary",
    ):
        from tssep.util.utils import stft_vad

        # squeeze: Remove mask idx dimension, only relevant for beamforming
        estimate = torch.squeeze(out.logit, dim=-3)
        assert self.target[0].isupper(), self.target
        if self.target not in ex:
            if self.target == "Vad":
                ex[self.target] = stft_vad(
                    ex[self.target.lower()],
                    model.fe.window_length,
                    model.fe.shift,
                    model.fe.fading,
                )
            else:
                ex[self.target] = model.fe.stft(ex[self.target.lower()])

        target = ex[self.target]
        return self(estimate, target)

    def update_summary(
        self,
        summary: ReviewSummary,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
    ):

        # Overwrite the mask. Add the mask with VAD information to tensorboard.
        target_vad = einops.repeat(
            self.prepare_target(ex[self.target]),
            "... spk time -> ... spk mask time freq",
            freq=40,
            mask=out.mask.shape[-3],
        )
        masks = torch.concat([target_vad, out.mask, target_vad], dim=-1)
        summary.add_mask_image(
            f"{model.enhancer.name}_mask",
            masks,
            rearrange="... spk mask time freq -> ... time (spk mask freq)",
            batch_first=True,
        )


class MSE(TimeDomain):
    def __init__(
        self,
        target: str = "speaker_reverberation_early_ch0",
        pit: bool = False,
    ):
        super().__init__(target=target, pit=pit)

    def loss_fn(self, estimate, target):
        """

        >>> _ = torch.manual_seed(0)
        >>> target = torch.rand((2, 10000))
        >>> estimate = target + 0.5 * torch.rand((2, 10000))
        >>> MSE(pit=False)(estimate, target)
        tensor(0.1673)
        >>> MSE(pit=False)(target, target)
        tensor(0.)
        """
        return pt.ops.mse_loss(estimate, target)


class MAE(TimeDomain):
    """
    Mean Absolute Error

    >>> _ = torch.manual_seed(0)
    >>> target = torch.rand((2, 10000))
    >>> estimate = target + 0.5 * torch.rand((2, 10000))
    >>> MAE(pit=False)(estimate, target)
    tensor(0.5018)
    >>> MAE(pit=False)(target, target)
    tensor(0.)
    """

    def __init__(
        self,
        target: str = "speaker_reverberation_early_ch0",
        pit: bool = False,
    ):
        super().__init__(target=target, pit=pit)
        self._loss_fn = torch.nn.L1Loss(reduction="none")

    def loss_fn(self, estimate, target):
        return self._loss_fn(estimate, target).mean(dim=-1).sum(dim=-1)


class LogMAE(TimeDomain):
    """
    Mean Absolute Error

    >>> _ = torch.manual_seed(0)
    >>> target = torch.rand((2, 10000))
    >>> estimate = target + 0.5 * torch.rand((2, 10000))
    >>> LogMAE(pit=False)(estimate, target)
    tensor(-0.2995)
    >>> LogMAE(pit=False)(target, target)
    tensor(-inf)
    >>> estimate[1, :] = 0
    >>> target[1, :] = 0
    >>> LogMAE(pit=False)(estimate, target)
    tensor(-0.5980)
    """

    def __init__(
        self,
        target: str = "speaker_reverberation_early_ch0",
        pit: bool = False,
    ):
        super().__init__(target=target, pit=pit)
        self._loss_fn = torch.nn.L1Loss(reduction="none")

    def loss_fn(self, estimate, target):
        return torch.log10(
            self._loss_fn(estimate, target).mean(dim=-1).sum(dim=-1)
        )


class FreqMSE(STFTDomain):
    def __init__(
        self,
        target: str = "Speaker_reverberation_early",
        pit: bool = False,
    ):
        super().__init__(target=target, pit=pit)

    def loss_fn(self, estimate, target):
        """

        >>> _ = torch.manual_seed(0)
        >>> target = torch.rand((2, 10000))
        >>> estimate = target + 0.5 * torch.rand((2, 10000))
        >>> FreqMSE(pit=False)(estimate, target)
        tensor(0.1673)
        >>> FreqMSE(pit=False)(target, target)
        tensor(0.)
        """
        return pt.ops.mse_loss(estimate, target)


class VADSigmoidBCE(LogitsSTFTDomain):
    def __init__(
        self,
        target: str = "Vad",
        pit: bool = False,
        magnitude_threshold: float = 0.05,
    ):
        super().__init__(target=target, pit=pit)
        assert 0 < magnitude_threshold < 1, magnitude_threshold
        self.magnitude_threshold = magnitude_threshold

    def loss_fn(self, estimate, target):
        """

        >>> _ = torch.manual_seed(0)
        >>> target = torch.rand((2, 100, 257))
        >>> estimate = target + 0.5 * torch.rand((2, 100, 257))
        >>> loss = VADSigmoidBCE(pit=False, target='Speaker_reverberation_early')
        >>> loss(estimate, target)
        tensor(0.3867)
        >>> loss.prepare_target(target).shape
        torch.Size([2, 100])
        >>> loss(((abs(target) > 0.05).float() - 0.5) * 500, target)
        tensor(0.)
        >>> loss(((abs(target) > 0.05).float() - 0.5) * 10, target)
        tensor(0.0111)
        >>> loss(((abs(target) > 0.05).float() - 0.5) * 1, target)
        tensor(0.4932)

        """
        return torch.nn.functional.binary_cross_entropy_with_logits(
            estimate, target, reduction="none"
        ).mean(
            # shape: (optional batch, speaker, time)
            dim=(
                -1,  # time dimension
                -2,  # speaker dimension
            )
        )

    def prepare_target(self, target, dtype=None):
        if self.target in ["vad", "Vad"]:
            return target

        if dtype is None:
            dtype = target.real.dtype
        if isinstance(target, torch.Tensor):
            target = abs(target).sum(axis=-1)
            target = target / torch.amax(target, dim=-1, keepdim=True)
            target = (target > self.magnitude_threshold).type(dtype)
            return target
        else:
            target = np.abs(target).sum(axis=-1)
            target = target / np.amax(target, axis=-1, keepdims=True)
            target = (target > self.magnitude_threshold).astype(dtype)
            return target

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        if not isinstance(target, torch.Tensor):
            target = torch.stack(target)

        if self.target not in ["vad", "Vad"]:
            assert estimate.shape == target.shape, (
                estimate.shape,
                target.shape,
            )

            assert estimate.ndim > 2, estimate.shape

            target = self.prepare_target(target)

        estimate = torch.mean(estimate, dim=-1)

        return super().forward(estimate, target)


class SignalAndVADSigmoidBCE(VADSigmoidBCE):
    # needs Net.explicit_vad = True
    def __init__(
        self,
        signal_loss: TimeDomain,
        target: str = "Vad",
        pit: bool = False,
        magnitude_threshold: float = 0.05,
    ):
        super().__init__(
            target=target, pit=pit, magnitude_threshold=magnitude_threshold
        )
        self.signal_loss = signal_loss

    def targets(self, lower=False, upper=False):
        return super().targets(
            lower=lower, upper=upper
        ) + self.signal_loss.targets(lower=lower, upper=upper)

    def from_ex_out(
        self,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
        summary: "pt.contrib.cb.summary.ReviewSummary",
    ):
        signal_loss = self.signal_loss.from_ex_out(ex, out, model, summary)

        from tssep.util.utils import stft_vad

        # squeeze: Remove mask idx dimension, only relevant for beamforming
        estimate = torch.squeeze(out.vad_logit[..., None], dim=-3)
        assert self.target[0].isupper(), self.target
        if self.target not in ex:
            if self.target == "Vad":
                ex[self.target] = stft_vad(
                    ex[self.target.lower()],
                    model.fe.window_length,
                    model.fe.shift,
                    model.fe.fading,
                )
            else:
                ex[self.target] = model.fe.stft(ex[self.target.lower()])

        target = ex[self.target]
        return self(estimate, target) + signal_loss

    def update_summary(
        self,
        summary: ReviewSummary,
        ex,
        out: "tssep.train.model.Model.ForwardOutput",
        model: "tssep.train.model.Model",
    ):

        # Overwrite the mask. Add the mask with VAD information to tensorboard.
        target_vad = einops.repeat(
            self.prepare_target(ex[self.target]),
            "... spk time -> ... spk mask time freq",
            freq=40,
            mask=out.mask.shape[-3],
        )
        estimate_vad = einops.repeat(
            out.vad_mask,
            "... spk mask time -> ... spk mask time freq",
            freq=40,
        )
        masks = torch.concat(
            [target_vad, estimate_vad, out.mask, estimate_vad, target_vad],
            dim=-1,
        )
        summary.add_mask_image(
            f"{model.enhancer.name}_mask",
            masks,
            rearrange="... spk mask time freq -> ... time (spk mask freq)",
            batch_first=True,
        )
