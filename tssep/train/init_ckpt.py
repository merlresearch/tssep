# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses
import typing
from pathlib import Path
from typing import Literal

import padertorch as pt
import torch

if typing.TYPE_CHECKING:
    import tssep.train.experiment


@dataclasses.dataclass
class InitCheckPoint(pt.Configurable):
    init_ckpt: "str | Path" = None
    strict: bool = True

    def load_model_state_dict(
        self, eg: "tssep.train.experiment.Experiment", ckpt
    ):
        ckpt = Path(ckpt)
        assert ckpt.exists(), ckpt
        state_dict = torch.load(str(ckpt), map_location="cpu")
        return eg.trainer.model.load_state_dict(
            state_dict["model"], strict=self.strict
        )

    def __call__(self, eg: "tssep.train.experiment.Experiment"):
        if self.init_ckpt is not None:
            self.load_model_state_dict(eg, self.init_ckpt)
        else:
            pass


@dataclasses.dataclass
class InitCheckPointVAD2Sep(InitCheckPoint):
    """
    Init a separation model from a VAD model:
     - Broadcast the parameters in the last layer.
    """

    bcast: tuple = (
        "mask_estimator.post_net.linear2.weight",
        "mask_estimator.post_net.linear2.bias",
    )
    mode: Literal["tile", "repeat"] = "repeat"  # numpy names
    # tile: [0, 1, 2] -> [0, 1, 2, 0, 1, 2]  # np.tile, torch.repeat
    # repeat: [0, 1, 2] ->  [0, 0, 1, 1, 2, 2]  # np.repeat, torch.repeat_interleave

    def load_model_state_dict(
        self, eg: "tssep.train.experiment.Experiment", ckpt
    ):
        """
        >>> init = InitCheckPointVAD2Sep('.../checkpoints/ckpt_2000.pth')  # doctest: +SKIP
        >>> eg = tssep.train.experiment.Experiment.from_file('.../config.yaml', 'eg')  # doctest: +SKIP
        >>> init(eg)  # doctest: +SKIP
        """
        ckpt = Path(ckpt)
        assert ckpt.exists(), ckpt
        state_dict = torch.load(str(ckpt), map_location="cpu")

        for k in self.bcast:
            shape = eg.trainer.model.get_parameter(k).shape
            p: torch.Tensor = state_dict["model"][k]
            assert len(p.shape) == len(shape), (p.shape, shape)
            assert self.mode == "repeat", f"ToDO: Implement {self.mode}"

            for i, (actual, desired) in enumerate(zip(p.shape, shape)):
                if actual == desired:
                    pass
                elif actual < desired:
                    assert desired % actual == 0, (
                        p.shape,
                        shape,
                        actual,
                        desired,
                    )
                    p = torch.repeat_interleave(p, desired // actual, dim=i)
                    state_dict["model"][k] = p
                else:
                    raise Exception(p.shape, shape, actual, desired)

        return eg.trainer.model.load_state_dict(
            state_dict["model"], strict=self.strict
        )
