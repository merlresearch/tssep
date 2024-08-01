# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses
import functools
import inspect
import os

import einops
import lazy_dataset
import numpy as np
import paderbox as pb  # noqa
import padertorch as pt
import torch
from padertorch.contrib.cb.summary import ReviewSummary

from tssep.data import DummyReader as Reader

# See tssep_data.data.reader_v2.Reader for a non-dummy reader.
from tssep.train import enhancer, feature_extractor, loss, net


def configurable_isinstance(cfg, classinfo):
    """
    Similar to `isinstance`, but assumes cfg is a config for a factory.

    It is assumed, that `isinstance(factory(...), classinfo) == subclass(factory, classinfo)` holds.
    For classes this is usually the case, unless someone writes a hack for a class, then this function will return
    the output of `subclass` and not from `isinstance` or an assert will warn the user.

    Args:
        cfg:
        classinfo:

    Returns:

    >>> a = {'factory': torch.nn.Linear, 'in_features': 2, 'out_features': 3}
    >>> b = {'factory': torch.nn.LazyLinear, 'out_features': 3}
    >>> for cfg in [a, b]:
    ...     print(cfg)
    ...     for classinfo, string in [(torch.nn.Linear, 'torch.nn.Linear'), (torch.nn.Module, 'torch.nn.Module'), (torch.nn.LazyLinear, 'torch.nn.LazyLinear')]:
    ...         print('    ', classinfo, isinstance(pt.Configurable.from_config(cfg), classinfo), configurable_isinstance(cfg, classinfo), configurable_isinstance(cfg, string))
    {'factory': <class 'torch.nn.modules.linear.Linear'>, 'in_features': 2, 'out_features': 3}
         <class 'torch.nn.modules.linear.Linear'> True True True
         <class 'torch.nn.modules.module.Module'> True True True
         <class 'torch.nn.modules.linear.LazyLinear'> False False False
    {'factory': <class 'torch.nn.modules.linear.LazyLinear'>, 'out_features': 3}
         <class 'torch.nn.modules.linear.Linear'> True True True
         <class 'torch.nn.modules.module.Module'> True True True
         <class 'torch.nn.modules.linear.LazyLinear'> True True True
    """
    assert "factory" in cfg, cfg
    factory = cfg["factory"]

    # Normalize factory and classinfo
    factory = pt.configurable.import_class(
        pt.configurable.class_to_str(factory)
    )
    classinfo = pt.configurable.import_class(
        pt.configurable.class_to_str(classinfo)
    )

    assert inspect.isclass(factory), factory
    assert inspect.isclass(classinfo), classinfo

    return issubclass(factory, classinfo)


class Model(pt.Model):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        """
        >>> pb.utils.pretty.pprint(Model.get_config())
        {'factory': 'tssep.train.model.Model',
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
          'pit': False}}

        """

        config["fe"] = dict(
            factory=feature_extractor.Log1pMaxNormAbsSTFT,
            size=1024,
            shift=256,
            window="hann",
        )
        config["reader"] = dict(
            factory=Reader,
        )

        config["enhancer"] = dict(
            factory=enhancer.Masking,
        )

        fe: feature_extractor.STFT = pt.Configurable.from_config(config["fe"])

        config["mask_estimator"] = dict(
            factory=net.MaskEstimator_v2,
            idim=fe.output_size,
            odim=fe.frequencies,
            nmask=(
                1
                if issubclass(
                    config["enhancer"]["factory"], (enhancer.Masking)
                )
                else 2
            ),
        )
        config["loss"] = dict(
            factory=loss.LogMAE,
        )

    def __init__(
        self,
        fe: feature_extractor.AbsSTFT,
        reader: Reader,
        mask_estimator: net.MaskEstimator_v2,
        enhancer: enhancer.Masking,
        loss: "loss.LogMAE",
    ):
        super().__init__()
        self.fe = fe
        self.reader = reader
        self.mask_estimator = mask_estimator
        self.enhancer = enhancer
        self.loss = loss

    def example_to_device(self, ex, device):
        memo = {}

        for k in {
            "Input",
            "observation",
            "auxInput",
            "framewise_embeddings",
            *self.loss.targets(lower=True),
            *self.loss.targets(),
        }:
            if k in ex:
                ex[k] = pt.data.example_to_device(ex[k], device, memo)

        return ex

    def prepare_dataset(
        self,
        dataset_name,
        device,
        training=False,  # Switch between torch train (training) and eval (validate and evaluation)
        review=True,  # Whether data review is necessary (e.g. the target of the loss) or optional (i.e. load only when available).
        batch_size=None,
        prefetch=True,
        reader=None,
        sort=False,  # Use for testrun. Starting with large examples will trigger OOM early.
        verbose=False,  # Changed for eval so the example contains more information
        load_keys=None,
    ):
        if reader is None:
            reader = self.reader

        if sort:

            def pre_load_apply(ds: lazy_dataset.Dataset):
                def get_num_samples(ex):
                    try:
                        return ex["end"] - ex["start"]
                    except KeyError:
                        pass
                    try:
                        return ex["num_samples"]["observation"]
                    except TypeError:
                        return ex["num_samples"]
                    except KeyError:
                        # TGb doesn't have num_sample for observation
                        # num_samples not nessesary, only need a
                        # rough estimate for the length to trigger an
                        # OOM early.
                        return max(ex["num_samples"]["original_source"])

                ds = ds.copy(freeze=True)
                ds = ds.sort(get_num_samples, reverse=True)
                return ds

        else:
            pre_load_apply = None

        if load_keys is None:
            load_keys = [
                "observation",
                *self.loss.targets(lower=True),
            ]

        ds = reader(
            dataset_name,
            pre_load_apply=pre_load_apply,
            load_keys=load_keys,
        )

        def prepare(ex):
            r = {}
            r["reference_channel"] = 0

            try:
                r["observation"] = ex["audio_data"]["observation"]
            except KeyError:
                # Special case for NativeKaldiTSVADTrain
                if "Input" in ex:
                    r["Input"] = ex["Input"]
                else:
                    raise

            for target_name in self.loss.targets():
                try:
                    target_name_lower = target_name.lower()
                    if target_name_lower in ex["audio_data"]:
                        target = ex["audio_data"][target_name_lower]
                        if (
                            isinstance(target, np.ndarray)
                            and len(target.shape) == 3
                        ):
                            target = target[:, r["reference_channel"]]
                        r[target_name_lower] = target

                    elif target_name in ["Vad"]:
                        r[target_name] = ex["audio_data"][target_name]
                    elif review:
                        raise Exception(
                            f"Either the reader has a bug and forgot to load {target_name.lower()!r} or you don't need the\n"
                            f"target signal from the loss. To disable this error, set the review flag to False."
                        )
                    else:
                        # LibriCSS usually doesn't have a target signal, hence ignore that this signal is unavailable.
                        pass
                except KeyError:
                    # Train: requires targets
                    # Val: might need targets
                    # Eval: targets not necessary and may not exist.
                    # Issue:
                    #     PyTorch knows only train and not train.
                    if self.training:
                        raise
                    else:
                        pass

            for k in [
                "example_id",
                "dataset",
                "gender",
                "auxInput",
                "vad",
                "framewise_embeddings",
                "framewise_embeddings_stride",
            ]:
                if k in ex:
                    r[k] = ex[k]

            if verbose:
                r["verbose"] = ex

            return r

        ds = ds.map(prepare)

        if training and not sort:
            ds = ds.shuffle(reshuffle=True)

        if batch_size is not None:
            ds = ds.batch(batch_size)
            ds = ds.map(self.collate_fn)

        if prefetch:
            backend = "t"
            # backend = 'mp'

            threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 6))
            ds = ds.prefetch(
                threads,
                threads * 2,
                catch_filter_exception=True,
                backend=backend,
            )

            if backend != "t":
                # Move data to main Process.
                ds = ds.prefetch(1, threads, backend="t")

        else:
            if training:
                ds = ds.catch()

        if device is not None:
            ds = ds.map(
                functools.partial(self.example_to_device, device=device)
            )

            if prefetch:
                # Only a small buffer for example_to_device.
                ds = ds.prefetch(1, 2)

        return ds

    def collate_fn(self, exs):
        ex = pt.data.utils.collate_fn(exs)

        try:
            k = "observation"
            ex[k] = np.array(ex[k])
        except Exception:
            assert "Input" in ex, ex.keys()
            k = "Input"
            ex[k] = np.array(ex[k])

        for k in [
            *self.loss.targets(),
            "vad",
            "framewise_embeddings",
            "auxInput",
        ]:
            k_lower = k.lower()
            if k_lower in ex:
                k = k_lower

            if k in ex:
                new = np.array(ex[k])
                if new.dtype != object:
                    ex[k] = new

        assert len(set(ex["reference_channel"])) == 1, ex["reference_channel"]
        # Selecting the reference channel inside a batch is much easier, when reference_channel is a scalar and
        # not a list with different reference channels for each entry in the batch
        ex["reference_channel"] = ex["reference_channel"][0]

        return ex

    def prepare_train_dataset(
        self,
        device,
        batch_size=None,
        prefetch=True,
        reader=None,
        sort=False,  # Use for testrun. Starting with large examples will trigger OOM early.
    ):
        """
        >>> model: Model = Model.new({'mask_estimator': {'units': 10, 'projs': 12}})
        >>> ds = model.prepare_train_dataset(device='cpu', prefetch=False)
        """
        return self.prepare_dataset(
            dataset_name=self.reader.train_dataset_name,
            training=True,
            device=device,
            batch_size=batch_size,
            prefetch=prefetch,
            reader=reader,
            sort=sort,
        )

    def prepare_validate_dataset(
        self,
        device,
        batch_size=None,
        prefetch=True,
        reader=None,
        sort=False,  # Use for testrun. Starting with large examples will trigger OOM early.
    ):
        """
        >>> model = Model.new({'mask_estimator': {'units': 10, 'projs': 12}})
        >>> ds = model.prepare_validate_dataset(device='cpu', prefetch=False)
        >>> ds  # doctest: +ELLIPSIS
              ListDataset(len=4)
            MapDataset(_pickle.loads)
          MapDataset(<function Model.prepare_dataset.<locals>.prepare at 0x...>)
        MapDataset(functools.partial(...)>, device='cpu'))
        >>> ex = next(iter(ds))
        >>> pb.utils.pretty.pprint(ex)  # doctest: +ELLIPSIS
        {'reference_channel': 0,
         'observation': tensor(shape=(1, 80000), dtype=float32),
         'speaker_reverberation_early_ch0': tensor(shape=(8, 80000), dtype=float32),
         'example_id': 'dummy_id_0',
         'dataset': 'validate',
         'auxInput': tensor(shape=(8, 100), dtype=float32)}
        """
        return self.prepare_dataset(
            dataset_name=self.reader.validate_dataset_name,
            training=False,
            device=device,
            batch_size=batch_size,
            prefetch=prefetch,
            reader=reader,
            sort=sort,
        )

    def prepare_eval_dataset(
        self,
        device,
        batch_size=None,
        prefetch=True,
        reader=None,
        sort=False,
        load_keys=None,
    ):
        if reader is None:
            reader = self.reader

        return self.prepare_dataset(
            dataset_name=reader.eval_dataset_name,
            training=False,
            device=device,
            batch_size=batch_size,
            prefetch=prefetch,
            review=False,
            reader=reader,
            sort=sort,
            verbose=True,
            load_keys=load_keys,
        )

    @dataclasses.dataclass
    class ForwardOutput:
        mask: torch.Tensor
        logit: torch.Tensor
        embedding: torch.Tensor = None
        stft_estimate: torch.Tensor = None
        time_estimate: torch.Tensor = None

        vad_mask: torch.Tensor = None
        vad_logit: torch.Tensor = None

    def forward(self, ex, feature_transform=None) -> ForwardOutput:
        """
        >>> pb.utils.pretty.pprint(Model.get_config())  # doctest: +ELLIPSIS
        {'factory': 'tssep.train.model.Model',
         'fe': {...},
         'reader': {...},
         'mask_estimator': {...},
         'enhancer': {'factory': 'tssep.train.enhancer.Masking'},
         'loss': {'factory': 'tssep.train.loss.LogMAE',
          'target': 'speaker_reverberation_early_ch0',
          'pit': False}}
        >>> model = Model.new({'mask_estimator': {'units': 10, 'projs': 12}})
        >>> ds = model.prepare_validate_dataset(device='cpu', prefetch=False)
        >>> ex = next(iter(ds))
        >>> pb.utils.pretty.pprint(model(ex))
        ForwardOutput(mask=tensor(shape=(8, 1, 316, 513), dtype=float32),
                      logit=tensor(shape=(8, 1, 316, 513), dtype=float32),
                      embedding=tensor(shape=(8, 1, 100), dtype=float32),
                      stft_estimate=tensor(shape=(8, 316, 513), dtype=complex64),
                      time_estimate=None,
                      vad_mask=None,
                      vad_logit=None)
        """

        ex["AuxInput"] = [a for a in ex["auxInput"]]

        if not isinstance(ex["reference_channel"], int):
            raise NotImplementedError(
                type(ex["reference_channel"]), ex["reference_channel"]
            )

        if "Input" in ex:
            pass
        elif "Observation" in ex:
            # At eval time the stft of the observation is precalculated, because WPE is sometimes used as preprocessor.
            ex["Input"] = self.fe.stft_to_feature(
                ex["Observation"][..., ex["reference_channel"], :, :]
            ).to(torch.float32)
        elif hasattr(self.fe, "stft"):
            ex["Observation"] = self.fe.stft(ex["observation"])
            ex["Input"] = self.fe.stft_to_feature(
                ex["Observation"][..., ex["reference_channel"], :, :]
            ).to(torch.float32)
        else:
            ex["Input"] = self.fe(
                ex["observation"][..., ex["reference_channel"], :]
            ).to(torch.float32)

        if feature_transform is not None:
            ex["Input"] = feature_transform(ex["Input"])

        ex = self.reader.data_hooks.pre_net(ex)

        me_out: net.Output = self.mask_estimator(
            ex["Input"],
            ex["AuxInput"],
        )

        if "Observation" in ex:
            stft_estimate = self.enhancer(me_out.mask, ex, self)
        else:
            assert isinstance(self.loss, loss.VADSigmoidBCE), type(self.loss)
            stft_estimate = None

        return self.ForwardOutput(
            mask=me_out.mask,
            logit=me_out.logit,
            vad_mask=me_out.vad_mask,
            vad_logit=me_out.vad_logit,
            embedding=me_out.embedding,
            stft_estimate=stft_estimate,
        )

    def review(self, ex, out: ForwardOutput):
        """
        >>> np.random.seed(0)
        >>> _ = torch.manual_seed(0)
        >>> pb.utils.pretty.pprint(Model.get_config())  # doctest: +ELLIPSIS
        {'factory': 'tssep.train.model.Model',
         'fe': {...},
         'reader': {...},
         'mask_estimator': {...},
         'enhancer': {'factory': 'tssep.train.enhancer.Masking'},
         'loss': {'factory': 'tssep.train.loss.LogMAE',
          'target': 'speaker_reverberation_early_ch0',
          'pit': False}}

        >>> model = Model.new({'mask_estimator': {'units': 10, 'projs': 12}})
        >>> print('Paremeters:', sum([p.numel() for p in model.parameters()]))
        Paremeters: 114038
        >>> ds = model.prepare_validate_dataset(device='cpu', prefetch=False, batch_size=2)
        >>> ex = next(iter(ds))
        >>> model.create_snapshot = False  # Probably the snapshots are to much for github actions

        >>> summary = model.review(ex, model(ex))
        >>> pb.utils.pretty.pprint(summary)
        ReviewSummary(
            prefix='',
            _data={'loss': tensor(1.4861, grad_fn=<SumBackward0>),
                   'scalars': {'validate_LogMAE': [array(0.74156505, dtype=float32),
                     array(0.744494, dtype=float32)]},
                   'histograms': {'hist_validate_LogMAE': [array(0.74156505, dtype=float32),
                     array(0.744494, dtype=float32)]}}
        )

        >>> torch.norm(ex['Input'])
        tensor(58.8257)
        >>> torch.std(ex['Input'])
        tensor(0.0960)
        >>> torch.amax(abs(ex['Input']))
        tensor(1.)

        >>> summary['loss'].backward()
        >>> for n, m in model.named_parameters():
        ...     print(n, torch.norm(m.grad))  # doctest: +ELLIPSIS
        mask_estimator.pre_net.net.0.weight_ih_l0 tensor(...)
        mask_estimator.pre_net.net.0.weight_hh_l0 tensor(...)
        mask_estimator.pre_net.net.0.bias_ih_l0 tensor(...)
        mask_estimator.pre_net.net.0.bias_hh_l0 tensor(...)
        mask_estimator.pre_net.net.0.weight_ih_l0_reverse tensor(...)
        mask_estimator.pre_net.net.0.weight_hh_l0_reverse tensor(...)
        mask_estimator.pre_net.net.0.bias_ih_l0_reverse tensor(...)
        mask_estimator.pre_net.net.0.bias_hh_l0_reverse tensor(...)
        mask_estimator.pre_net.net.1.weight tensor(...)
        mask_estimator.pre_net.net.1.bias tensor(...)
        mask_estimator.post_net.birnn0.net.0.weight_ih_l0 tensor(...)
        mask_estimator.post_net.birnn0.net.0.weight_hh_l0 tensor(...)
        mask_estimator.post_net.birnn0.net.0.bias_ih_l0 tensor(...)
        mask_estimator.post_net.birnn0.net.0.bias_hh_l0 tensor(...)
        mask_estimator.post_net.birnn0.net.0.weight_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn0.net.0.weight_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn0.net.0.bias_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn0.net.0.bias_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn0.net.1.weight tensor(...)
        mask_estimator.post_net.birnn0.net.1.bias tensor(...)
        mask_estimator.post_net.birnn1.net.0.weight_ih_l0 tensor(...)
        mask_estimator.post_net.birnn1.net.0.weight_hh_l0 tensor(...)
        mask_estimator.post_net.birnn1.net.0.bias_ih_l0 tensor(...)
        mask_estimator.post_net.birnn1.net.0.bias_hh_l0 tensor(...)
        mask_estimator.post_net.birnn1.net.0.weight_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn1.net.0.weight_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn1.net.0.bias_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn1.net.0.bias_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn1.net.1.weight tensor(...)
        mask_estimator.post_net.birnn1.net.1.bias tensor(...)
        mask_estimator.post_net.birnn2.net.0.weight_ih_l0 tensor(...)
        mask_estimator.post_net.birnn2.net.0.weight_hh_l0 tensor(...)
        mask_estimator.post_net.birnn2.net.0.bias_ih_l0 tensor(...)
        mask_estimator.post_net.birnn2.net.0.bias_hh_l0 tensor(...)
        mask_estimator.post_net.birnn2.net.0.weight_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn2.net.0.weight_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn2.net.0.bias_ih_l0_reverse tensor(...)
        mask_estimator.post_net.birnn2.net.0.bias_hh_l0_reverse tensor(...)
        mask_estimator.post_net.birnn2.net.1.weight tensor(...)
        mask_estimator.post_net.birnn2.net.1.bias tensor(...)
        mask_estimator.post_net.linear2.weight tensor(...)
        mask_estimator.post_net.linear2.bias tensor(...)

        >>> from padertorch.contrib.cb.track import track, tracker_list, ShapeTracker, ParameterTracker
        >>> with track(model, tracker_list(ShapeTracker, ParameterTracker), leaf_types=(net.AuxNet, net.RNNP_packed)) as trackers:
        ...     summary = model.review(ex, model(ex))
        >>> print(trackers)
                                                  input                            output        #Params
          0 Model:               ({'reference_channel': '?', 'observation ->          ?                0
                                 ': [2, 1, 80000], 'speaker_reverberation
                                 _early_ch0': [2, 8, 80000], 'example_id'
                                 : ['?', '?'], 'dataset': ['?', '?'], 'au
                                 xInput': [2, 8, 100], 'AuxInput': [[8, 1
                                 00], [8, 100]], 'Observation': [2, 1, 31
                                    6, 513], 'Input': [2, 316, 513]},)
          1   MaskEstimator_v2:   ([2, 316, 513], [[8, 100], [8, 100]])   ->          ?                0
          2     RNNP_packed:                 ([2, 316, 513],)             ->    [2, 316, 513]     52_773
          3     Sequential:                ([2, 8, 316, 613],)            -> [2, 8, 1, 316, 513]       0
          4       RNNP_packed:             ([2, 8, 316, 613],)            ->   [2, 8, 316, 12]    50_252
          5       Dropout:                  ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]         0
          6       Tanh:                     ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]         0
          7       RNNP_packed:              ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]     2_172
          8       Dropout:                  ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]         0
          9       Tanh:                     ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]         0
         10       RNNP_packed:              ([2, 8, 316, 12],)            ->   [2, 8, 316, 12]     2_172
         11       Linear:                   ([2, 8, 316, 12],)            ->  [2, 8, 316, 513]     6_669
         12       Rearrange:               ([2, 8, 316, 513],)            -> [2, 8, 1, 316, 513]       0
         13     Sigmoid:                  ([2, 8, 1, 316, 513],)          -> [2, 8, 1, 316, 513]       0
         14 LogMAE:                   ([2, 8, 80000], [2, 8, 80000])      ->         [2]               0
         15   L1Loss:                 ([2, 8, 80000], [2, 8, 80000])      ->    [2, 8, 80000]          0

        """

        masks = out.mask
        stft_estimate = out.stft_estimate

        if self.training:
            summary = ReviewSummary()
        else:
            summary = ReviewSummary()

        if hasattr(self.fe, "istft") and "observation" in ex:
            out.time_estimate = self.fe.istft(
                stft_estimate, num_samples=ex["observation"].shape[-1]
            )

        loss_value: torch.Tensor
        loss_value = self.loss.from_ex_out(ex, out, self, summary)

        summary.add_to_loss(loss_value.sum())

        with torch.no_grad():
            if loss_value.ndim == 0:
                has_batch_dim = False
                summary.add_scalar(
                    f'{ex["dataset"]}_{self.loss.name}', loss_value
                )
                summary.add_histogram(
                    f'hist_{ex["dataset"]}_{self.loss.name}', loss_value
                )
            elif loss_value.ndim == 1:
                has_batch_dim = True
                for dataset_name, lv in zip(ex["dataset"], loss_value):
                    summary.add_scalar(f"{dataset_name}_{self.loss.name}", lv)
                    summary.add_histogram(
                        f"hist_{dataset_name}_{self.loss.name}", lv
                    )
            else:
                raise NotImplementedError(
                    loss_value.ndim, loss_value.shape, ex["dataset"]
                )

            if self.create_snapshot:
                if out.time_estimate is not None:
                    for i, e in enumerate(out.time_estimate):
                        summary.add_audio(
                            f"{self.enhancer.name}_audio_est_{i}",
                            e,
                            sampling_rate=self.reader.sample_rate,
                            batch_first=True,
                        )

                if "observation" in ex:
                    summary.add_audio(
                        f"{self.enhancer.name}_audio_observation",
                        ex["observation"][ex["reference_channel"]],
                        sampling_rate=self.reader.sample_rate,
                        batch_first=True,
                    )

                if "Observation" in ex:
                    summary.add_stft_image(
                        f"{self.enhancer.name}_Observation",
                        ex["Observation"][ex["reference_channel"]],
                        batch_first=True,
                    )

                dataset_name = ex["dataset"]
                if has_batch_dim:
                    dataset_name = dataset_name[0]

                summary.add_mask_image(
                    f"{dataset_name}_{self.enhancer.name}_mask",
                    masks[0] if has_batch_dim else masks,
                    rearrange="spk mask time freq -> time (spk mask freq)",
                    batch_first=None,
                )
                if out.stft_estimate is not None:
                    summary.add_stft_image(
                        f"{self.enhancer.name}_stft_estimate",
                        out.stft_estimate,
                        rearrange="... spk time freq -> ... time (spk freq)",
                        batch_first=True,
                    )

                for target_name in self.loss.targets(upper=True):
                    if target_name == "Vad":
                        target = einops.repeat(
                            ex[target_name], "... -> ... freq", freq=40
                        )
                    elif target_name in ex:
                        target = ex[target_name]
                    else:
                        target = self.fe.stft(ex[target_name.lower()])

                    summary.add_stft_image(
                        f"{self.enhancer.name}_target_{target_name}",
                        target,
                        rearrange="... spk time freq -> ... time (spk freq)",
                        batch_first=True,
                    )

                self.loss.update_summary(summary, ex, out, self)

        return summary
