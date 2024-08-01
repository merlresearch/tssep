# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses
import functools
import os
from pathlib import Path

import paderbox as pb  # noqa
import padertorch as pt
import torch

import tssep.train.model
from tssep.train.init_ckpt import InitCheckPoint


@dataclasses.dataclass
class Experiment(pt.Configurable):
    """
    >>> eg = Experiment.new({'trainer': {'storage_dir': ''}})
    >>> eg.trainer.model  # doctest: +ELLIPSIS
    Model(
      size=ModelParameterSize(...)
      (mask_estimator): MaskEstimator_v2(
        combination='cat',
        (pre_net): RNNP_packed(
          (net): ModuleList(
            (0): LSTM(513, 300, batch_first=True, bidirectional=True)
            (1): Linear(in_features=600, out_features=513, bias=True)
          )
        )
        (post_net): Sequential(
          (birnn0): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(613, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=320, bias=True)
            )
          )
          (dropout0): Dropout(p=0, inplace=False)
          (activation0): Tanh()
          (birnn1): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(320, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=320, bias=True)
            )
          )
          (dropout1): Dropout(p=0, inplace=False)
          (activation1): Tanh()
          (birnn2): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(320, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=320, bias=True)
            )
          )
          (linear2): Linear(in_features=320, out_features=513, bias=True)
          (rearrange2): Rearrange('... spk time (mask freq) -> ... spk mask time freq', mask=1)
        )
        (final_activation): Sigmoid()
      )
      (enhancer): Masking()
      (loss): LogMAE(
        target='speaker_reverberation_early_ch0', pit=False
        (_loss_fn): L1Loss()
      )
    )
    >>> pb.utils.pretty.pprint(Experiment.get_config({'trainer': {'storage_dir': ''}}))
    {'factory': 'tssep.train.experiment.Experiment',
     'trainer': {'factory': 'padertorch.train.trainer.Trainer',
      'model': {'factory': 'tssep.train.model.Model',
       'fe': {'factory': 'tssep.train.feature_extractor.Log1pMaxNormAbsSTFT',
        'size': 1024,
        'shift': 256,
        'window_length': 1024,
        'pad': True,
        'fading': True,
        'output_size': 513,
        'window': 'hann',
        'statistics_axis': 'tf'},
       'reader': {'factory': 'tssep.data.DummyReader',
        'train_dataset_name': 'train',
        'validate_dataset_name': 'validate',
        'domain_adaptation_src_dataset_name': 'validate',
        'eval_dataset_name': 'eval',
        'sample_rate': 16000,
        'aux_size': 100,
        'train_examples': 10},
       'mask_estimator': {'factory': 'tssep.train.net.MaskEstimator_v2',
        'idim': 513,
        'odim': 513,
        'layers': 3,
        'units': 300,
        'projs': 320,
        'dropout': 0,
        'nmask': 1,
        'pre_net': 'RNNP',
        'aux_net': None,
        'aux_net_output_size': 100,
        'combination': 'cat',
        'ts_vad': False,
        'output_resolution': 'tf',
        'random_speaker_order': True,
        'num_averaged_permutations': 1,
        'input_normalizer': None,
        'aux_normalizer': None,
        'explicit_vad': False},
       'enhancer': {'factory': 'tssep.train.enhancer.Masking'},
       'loss': {'factory': 'tssep.train.loss.LogMAE',
        'target': 'speaker_reverberation_early_ch0',
        'pit': False}},
      'storage_dir': '',
      'optimizer': {'factory': 'padertorch.train.optimizer.Adam',
       'gradient_clipping': 10,
       'lr': 0.001,
       'betas': [0.9, 0.999],
       'eps': 1e-08,
       'weight_decay': 0,
       'amsgrad': False},
      'loss_weights': None,
      'summary_trigger': [83, 'iteration'],
      'checkpoint_trigger': [1000, 'iteration'],
      'stop_trigger': [416666, 'iteration'],
      'virtual_minibatch_size': 12},
     'train_batchsize': None,
     'validation_batchsize': None,
     'init_ckpt': {'factory': 'tssep.train.init_ckpt.InitCheckPoint',
      'init_ckpt': None,
      'strict': True},
     'init_ckpt_strict': True}
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):

        virtual_minibatch_size = 12

        config["trainer"] = {
            "factory": pt.Trainer,
            "model": {"factory": tssep.train.model.Model},
            "summary_trigger": [1000 // virtual_minibatch_size, "iteration"],
            "checkpoint_trigger": [
                12000 // virtual_minibatch_size,
                "iteration",
            ],
            "stop_trigger": [5_000_000 // virtual_minibatch_size, "iteration"],
            "virtual_minibatch_size": virtual_minibatch_size,
            "optimizer": {
                "factory": pt.train.optimizer.Adam,
                "gradient_clipping": 10,
            },
        }

    trainer: pt.Trainer = dataclasses.field(
        default_factory=functools.partial(
            pt.Trainer,
            storage_dir=".",
        )
    )
    train_batchsize: int = None
    validation_batchsize: int = None
    init_ckpt: "tssep.train.init_ckpt.InitCheckPoint" = dataclasses.field(
        default_factory=InitCheckPoint
    )
    init_ckpt_strict: bool = True

    @functools.cached_property
    def data_device(self):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # For multple GPUs move the data inside the trainer.
                device = "cpu"
            else:
                device = 0
        else:
            device = "cpu"
        return device

    @functools.cached_property
    def device(self):
        if torch.cuda.is_available():
            assert torch.cuda.device_count() == 1, (
                torch.cuda.device_count(),
                "Only one GPU is supported. Set CUDA_VISIBLE_DEVICES to reduce the number of GPUs.",
            )
            if torch.cuda.device_count() > 1:
                device = tuple(range(torch.cuda.device_count()))
            else:
                device = 0
        else:
            device = "cpu"
        return device

    def device_count(self):
        device = self.device
        if isinstance(device, tuple):
            return len(device)
        else:
            return 1

    def load_model_state_dict(self, ckpt, strict=True):
        ckpt = Path(ckpt)
        assert ckpt.exists(), ckpt
        state_dict = torch.load(str(ckpt), map_location="cpu")
        return self.trainer.model.load_state_dict(
            state_dict["model"], strict=strict
        )

    def add_log_files(
        self,
        **kwargs,
    ):
        log_dir = self.trainer.storage_dir / "log"
        log_dir.mkdir(exist_ok=True, parents=True)
        (log_dir / "experiment.txt").write_text(str(self))
        (log_dir / "model.txt").write_text(str(self.trainer.model))
        for k, v in kwargs.items():
            (log_dir / f"{k}.txt").write_text(str(v))

    def train(self):
        print(
            f'Start training $CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}'
        )

        checkpoint_path = self.trainer.checkpoint_dir / "ckpt_latest.pth"
        resume = checkpoint_path.is_file()

        if resume:
            pass
        else:
            self.init_ckpt(self)

        self.trainer.model: "tssep.train.model.Model"

        from padertorch.contrib.cb.track import (
            GPUMemTracker,
            GPUTotPostMemTracker,
            GPUTotPreMemTracker,
            OBackwardMemTracker,
            ParameterTracker,
            ShapeTracker,
            TimeTracker,
            track,
            tracker_list,
        )

        with track(
            self.trainer.model,
            tracker_list(
                ShapeTracker,
                ParameterTracker,
                TimeTracker,
                GPUMemTracker,
                GPUTotPreMemTracker,
                GPUTotPostMemTracker,
                OBackwardMemTracker,
            ),
        ) as trackers:

            test_run_train_ds = self.trainer.model.prepare_train_dataset(
                self.data_device,
                prefetch=False,
                sort=True,
                batch_size=1,  # Caching intermediate results is expensive so we reduce the batch size for the test_run
                # batch_size=self.train_batchsize
            )
            test_run_validation_ds = (
                self.trainer.model.prepare_validate_dataset(
                    self.data_device,
                    prefetch=False,
                    sort=True,
                    batch_size=self.validation_batchsize,
                )
            )

            self.add_log_files(
                test_run_train_dataset=repr(test_run_train_ds),
                test_run_validation_dataset=repr(test_run_validation_ds),
            )

            print("Test run")
            self.trainer.test_run(
                test_run_train_ds,
                test_run_validation_ds,
                deterministic_atol=1e10,
                deterministic_rtol=1e10,
                loss_atol=1e10,
                loss_rtol=1e10,
                # Large tolerances needed due to shuffled utterances in speaker-specific layers.
                virtual_minibatch_size=1,
            )
            del test_run_train_ds, test_run_validation_ds
            print("Finished test run")

        validation_ds = self.trainer.model.prepare_validate_dataset(
            self.data_device,
            prefetch=True,
            batch_size=self.validation_batchsize,
        )
        self.trainer.register_validation_hook(
            validation_ds, max_checkpoints=None
        )

        train_ds = self.trainer.model.prepare_train_dataset(
            self.data_device, prefetch=True, batch_size=self.train_batchsize
        )

        self.add_log_files(
            train_dataset=repr(train_ds),
            validation_dataset=repr(validation_ds),
            trackers=str(trackers),
        )

        if self.device_count() > 1:
            self.trainer.model = torch.nn.DataParallel(self.trainer.model)

        print(
            "Train",
            f'$CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}',
        )
        self.trainer.train(train_ds, device=self.device, resume=resume)
