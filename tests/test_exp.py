# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import tempfile
from pathlib import Path

import paderbox as pb

from tssep.exp import run_tssep, run_tsvad
from tssep.train.experiment import Experiment


def reduce_parameters(config, *updates):
    """
    Reduce the parameters to speedup the tests.

    """
    flat = pb.utils.nested.FlatView(config)

    changes = {}
    changes["eg.trainer.summary_trigger"] = [1, "iteration"]
    changes["eg.trainer.checkpoint_trigger"] = [1, "iteration"]
    changes["eg.trainer.stop_trigger"] = [2, "iteration"]
    changes["eg.trainer.model.mask_estimator.units"] = 2
    changes["eg.trainer.model.mask_estimator.projs"] = 3
    changes["eg.trainer.model.mask_estimator.num_averaged_permutations"] = 1
    # sample_rate is multiplied by 5 to obtain the number of samples in DummyReader
    changes["eg.trainer.model.reader.sample_rate"] = 32

    # Changing the stft size need many changes in the config
    changes["eg.trainer.model.fe.size"] = 64
    changes["eg.trainer.model.fe.shift"] = 32
    changes["eg.trainer.model.fe.window_length"] = changes[
        "eg.trainer.model.fe.size"
    ]

    changes["eg.trainer.model.fe.fe1.size"] = changes[
        "eg.trainer.model.fe.size"
    ]
    changes["eg.trainer.model.fe.fe1.shift"] = changes[
        "eg.trainer.model.fe.shift"
    ]
    changes["eg.trainer.model.fe.fe1.window_length"] = changes[
        "eg.trainer.model.fe.window_length"
    ]
    changes["eg.trainer.model.fe.fe1.output_size"] = (
        40  # doesn't change, still 40 MFCCs
    )

    changes["eg.trainer.model.fe.fe2.size"] = changes[
        "eg.trainer.model.fe.size"
    ]
    changes["eg.trainer.model.fe.fe2.shift"] = changes[
        "eg.trainer.model.fe.shift"
    ]
    changes["eg.trainer.model.fe.fe2.window_length"] = changes[
        "eg.trainer.model.fe.window_length"
    ]
    changes["eg.trainer.model.fe.fe2.output_size"] = (
        changes["eg.trainer.model.fe.size"] // 2 + 1
    )

    changes["eg.trainer.model.fe.output_size"] = (
        changes["eg.trainer.model.fe.fe1.output_size"]
        + changes["eg.trainer.model.fe.fe2.output_size"]
    )

    changes["eg.trainer.model.mask_estimator.idim"] = changes[
        "eg.trainer.model.fe.output_size"
    ]
    changes["eg.trainer.model.mask_estimator.odim"] = (
        changes["eg.trainer.model.fe.size"] // 2 + 1
    )

    changes["eg.trainer.model.reader.aux_size"] = (
        changes["eg.trainer.model.fe.size"] // 2 + 1
    )
    changes["eg.trainer.model.mask_estimator.aux_net_output_size"] = changes[
        "eg.trainer.model.reader.aux_size"
    ]

    for k, v in changes.items():
        _ = flat[k]  # check that the key exists
        flat[k] = v

    for u in updates:
        u_flat = pb.utils.nested.FlatView(u)
        for k, v in u_flat.items():
            try:
                flat[k] = v
            except KeyError:
                flat[k[:-1]] = {k[-1]: v}

    return config


def test_run_tsvad():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = run_tsvad._cwd / "init_cfg_common.yaml"
        config = pb.io.load(config_file)

        config = reduce_parameters(
            config,
            pb.io.load(
                run_tsvad._cwd / "init_cfg_tsvad.yaml"
            ),  # update the config as done by sacred in the run script
            {"eg": {"trainer": {"storage_dir": tmpdir / "tsvad"}}},
        )
        from tssep.train.experiment import Experiment

        eg: Experiment = Experiment.from_config(config["eg"])
        eg.train()


def test_run_tssep():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = run_tssep._cwd / "init_cfg_common.yaml"
        config = pb.io.load(config_file)

        config = reduce_parameters(
            config,
            pb.io.load(
                run_tssep._cwd / "init_cfg_tssep.yaml"
            ),  # update the config as done by sacred in the run script
            {"eg": {"trainer": {"storage_dir": tmpdir / "tssep"}}},
        )

        eg: Experiment = Experiment.from_config(config["eg"])
        eg.train()


def test_run_tsvad_tssep():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = run_tsvad._cwd / "init_cfg_common.yaml"
        config = pb.io.load(config_file)
        config = reduce_parameters(config)

        config_file = tmpdir / "init_cfg_common.yaml"
        pb.io.dump(config, config_file)

        configs = [
            config_file,
            run_tsvad._cwd / "init_cfg_tsvad.yaml",
        ]

        storage_dir = tmpdir / "tsvad"
        run_tsvad.main(configs=configs, storage_dir=storage_dir)

        configs = [
            config_file,
            run_tssep._cwd / "init_cfg_tssep.yaml",
        ]

        checkpoint = storage_dir / "checkpoints/ckpt_best_loss.pth"
        storage_dir = tmpdir / "tssep"
        run_tssep.main(
            configs=configs, storage_dir=storage_dir, checkpoint=checkpoint
        )
