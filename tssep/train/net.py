# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import collections
import dataclasses

import einops.layers.torch
import numpy as np
import padertorch as pt
import torch
import torch.nn.functional
from einops import rearrange
from paderbox.utils.iterable import zip

from tssep.train.rnnp import RNNP_packed


class Linear(pt.Configurable, torch.nn.Module):
    def __init__(
        self,
        idim,
        odim,
        bias=True,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.bias = bias
        self.net = torch.nn.Linear(idim, odim, bias=bias)

    def forward(
        self,
        AuxInput,  # List of shape (time x freq) or "list of lists of shape (time x freq)"
        Input,  # (time x freq)  $ Ignored
        batched=False,
    ):
        if isinstance(AuxInput, (tuple, list)):
            if isinstance(AuxInput[0], (tuple, list)):
                AuxInput = torch.stack([torch.stack(i) for i in AuxInput])
            else:
                AuxInput = torch.stack(AuxInput)
        return self.net(AuxInput)


class AuxNet(pt.Configurable, torch.nn.Module):
    """
    Speakerbeam style auxiliary network.

    A few MLP layers followed by mean reduction over the time axis to obtain
    one embedding per speaker.
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config["odim"] is None:
            config["odim"] = config["idim"]

    def __init__(
        self,
        idim,
        odim=None,
        normalizer: "InstanceNorm" = None,
    ):
        """
        >>> from paderbox.utils.pretty import pprint
        >>> AuxNet.get_config({'idim': 513})
        {'factory': 'tssep.train.net.AuxNet', 'idim': 513, 'odim': 513, 'normalizer': None}
        >>> AuxNet(513)
        AuxNet(
          (net): Sequential(
            (0): Linear(in_features=513, out_features=513, bias=True)
            (1): ReLU()
            (2): Linear(in_features=513, out_features=513, bias=True)
            (3): ReLU()
            (4): Linear(in_features=513, out_features=513, bias=True)
          )
        )
        >>> AuxNet(10)([torch.randn((15, 10)), torch.randn((20, 10))], None).shape
        torch.Size([2, 10])
        >>> AuxNet(10)([[torch.randn((15, 10)), torch.randn((20, 10))], [torch.randn((16, 10)), torch.randn((21, 10))], [torch.randn((14, 10)), torch.randn((19, 10))]], None, batched=True).shape
        torch.Size([3, 2, 10])

        >>> cfg = AuxNet.get_config({'idim': 10, 'normalizer': {'factory': InstanceNorm}})
        >>> pprint(cfg)
        {'factory': 'tssep.train.net.AuxNet',
         'idim': 10,
         'odim': 10,
         'normalizer': {'factory': 'tssep.train.net.InstanceNorm',
          'dim': -1,
          'unbiased': False}}
        >>> aux_net = AuxNet.from_config(cfg)
        >>> aux_net
        AuxNet(
          (net): Sequential(
            (0): InstanceNorm(dim=-1, unbiased=False)
            (1): Linear(in_features=10, out_features=10, bias=True)
            (2): ReLU()
            (3): Linear(in_features=10, out_features=10, bias=True)
            (4): ReLU()
            (5): Linear(in_features=10, out_features=10, bias=True)
          )
        )
        >>> aux_net([torch.randn((15, 10)), torch.randn((20, 10))], None).shape
        torch.Size([2, 10])
        >>> aux_net([[torch.randn((15, 10)), torch.randn((20, 10))], [torch.randn((16, 10)), torch.randn((21, 10))], [torch.randn((14, 10)), torch.randn((19, 10))]], None, batched=True).shape
        torch.Size([3, 2, 10])
        """
        super().__init__()
        if odim is None:
            odim = idim
        elif idim == odim:
            pass
        else:
            raise NotImplementedError(odim, idim)
        self.odim = odim

        self.net = Sequential(
            *[e for e in [normalizer] if e is not None],
            torch.nn.Linear(idim, idim),
            torch.nn.ReLU(),
            torch.nn.Linear(idim, idim),
            torch.nn.ReLU(),
            torch.nn.Linear(idim, idim),
        )

    def forward(
        self,
        AuxInput,  # List of shape (time x freq) or "list of lists of shape (time x freq)"
        Input,  # (time x freq)  $ Ignored
        batched=False,
    ):
        h = AuxInput

        if batched:
            assert len({len(a) for a in AuxInput}) == 1, [
                len(a) for a in AuxInput
            ]
            # Flatten
            h = [e for a in AuxInput for e in a]

        ilens = [len(e) for e in h]
        h = torch.nn.utils.rnn.pad_sequence(h, batch_first=True)

        h = self.net(h)

        h = padded_sequence_reduction(
            h, sequence_lengths=ilens, sequence_axis=1, batch_axis=0, op="mean"
        )

        if batched:
            h = rearrange(
                h,
                "(batch speaker) feature -> batch speaker feature",
                batch=len(AuxInput),
            )

        return h


class Itemgetter(torch.nn.Module):
    """
    >>> Itemgetter(0)
    Itemgetter(0)
    >>> Itemgetter(0)([1, 2, 3])
    1
    """

    def __init__(self, item):
        super().__init__()
        self.item = item

    def extra_repr(self):
        return f"{self.item}"

    def forward(self, x):
        return x[self.item]


class SequentialFirstArgsKwargs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for i, module in enumerate(self):
            if i == 0:
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


class Sequential(torch.nn.Sequential):
    """
    Idea: Delete intermediate values as fast as possible to minimize the
    memory consumption.

    >>> import gc
    >>> class Foo:
    ...     def __init__(self, name):
    ...         self.name = name
    ...         print('Init', self.name)
    ...     def __del__(self):
    ...         print('Del', self.name)
    >>> class Bar(torch.nn.Module):
    ...     def forward(self, input):
    ...         new = input.name + 1
    ...         del input
    ...         gc.collect()
    ...         return Foo(new)
    >>> Sequential(Bar(), Bar(), Bar())(Foo(1)); gc.collect();  # doctest: +ELLIPSIS
    Init 1
    Init 2
    Init 3
    Del 2
    Init 4
    Del 3
    Del 1
    <tssep.train.net.Foo object at 0x...>
    Del 4
    0
    >>> torch.nn.Sequential(Bar(), Bar(), Bar())(Foo(1)); gc.collect();  # doctest: +ELLIPSIS
    Init 1
    Init 2
    Init 3
    Del 2
    Init 4
    Del 3
    Del 1
    <tssep.train.net.Foo object at 0x...>
    Del 4
    0

    """

    def forward(self, input):
        for module in self:
            input = [input]
            input = module(input.pop())
        return input


@dataclasses.dataclass
class Output:
    mask: torch.Tensor
    logit: torch.Tensor
    embedding: torch.Tensor = None

    vad_mask: torch.Tensor = None
    vad_logit: torch.Tensor = None


class InstanceNorm(torch.nn.Module):
    """
    >>> np.random.seed(0)
    >>> t = torch.tensor(np.array([np.random.randn(50) * 5 - 5, np.random.randn(50) * 0.5 + 100]))
    >>> torch.std_mean(t, dim=-1)
    (tensor([5.6847, 0.4379], dtype=torch.float64), tensor([-4.2972, 99.9895], dtype=torch.float64))
    >>> norm = InstanceNorm(dim=-1)
    >>> norm
    InstanceNorm(dim=-1, unbiased=False)
    >>> torch.std_mean(norm(t))
    (tensor(1.0050, dtype=torch.float64), tensor(-2.5235e-14, dtype=torch.float64))
    """

    def __init__(self, dim=-1, unbiased=False):
        super().__init__()
        self.dim = dim

        # Default based on https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html#torch.nn.InstanceNorm1d
        self.unbiased = unbiased

    def extra_repr(self):
        import inspect

        sig = inspect.signature(self.__class__)
        return ", ".join(
            [
                f"{p.name}={getattr(self, p.name)!r}"
                for p in sig.parameters.values()
            ]
        )

    def forward(self, x):
        std, mean = torch.std_mean(
            x, dim=self.dim, unbiased=self.unbiased, keepdim=True
        )
        return (x - mean) / std


class InstanceNorm_v2(torch.nn.Module):
    """
    >>> np.random.seed(0)
    >>> t = torch.tensor(np.array([np.random.randn(10) * 5 - 5, np.random.randn(10) * 0.5 + 100]))
    >>> t.shape
    torch.Size([2, 10])
    >>> InstanceNorm(dim=-1)(t)
    tensor([[ 1.0610, -0.3494,  0.2489,  1.5540,  1.1680, -1.7737,  0.2193, -0.9197,
             -0.8699, -0.3386],
            [-0.3811,  1.5646,  0.5352, -0.4143,  0.0642, -0.0995,  1.6237, -0.8996,
             -0.1301, -1.8633]], dtype=torch.float64)
    >>> InstanceNorm_v2(-1, -1)(t)
    tensor([[ 1.0610, -0.3494,  0.2489,  1.5540,  1.1680, -1.7737,  0.2193, -0.9197,
             -0.8699, -0.3386],
            [-0.3811,  1.5646,  0.5352, -0.4143,  0.0642, -0.0995,  1.6237, -0.8996,
             -0.1301, -1.8633]], dtype=torch.float64)
    """

    def __init__(self, mean_dim=-1, norm_dim=-1):
        super().__init__()
        self.mean_dim = mean_dim
        self.norm_dim = norm_dim

    def extra_repr(self):
        import inspect

        sig = inspect.signature(self.__class__)
        return ", ".join(
            [
                f"{p.name}={getattr(self, p.name)!r}"
                for p in sig.parameters.values()
            ]
        )

    def forward(self, x):
        mean = torch.mean(x, dim=self.mean_dim, keepdim=True)
        x = x - mean

        norm = torch.linalg.norm(x, dim=self.norm_dim, keepdim=True)
        norm = norm / np.sqrt(x.shape[self.norm_dim])
        x = x / norm

        return x


class MaskEstimator_v2(pt.Configurable, torch.nn.Module):
    """

    Changes to v1 (At time of creating v2):
     - Use custom RNNP: Verbose representer, disable dropout at eval time and less mem copies.
     - Create 3 RNNPs with layer parameter == 1, instead of 1 RNNP with layer parameter == 3

    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """
        >>> from paderbox.utils.pretty import pprint
        >>> pprint(MaskEstimator_v2.get_config({'combination': 'cat'}))
        {'factory': 'tssep.train.net.MaskEstimator_v2',
         'idim': 80,
         'odim': None,
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
         'explicit_vad': False}
        >>> me = MaskEstimator_v2.new({'combination': 'mul'})
        >>> me
        MaskEstimator_v2(
          combination='mul',
          (pre_net): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(80, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=80, bias=True)
            )
          )
          (post_net): Sequential(
            (birnn0): RNNP_packed(
              (net): ModuleList(
                (0): LSTM(80, 300, batch_first=True, bidirectional=True)
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
            (linear2): Linear(in_features=320, out_features=80, bias=True)
            (rearrange2): Rearrange('... spk time (mask freq) -> ... spk mask time freq', mask=1)
          )
          (final_activation): Sigmoid()
        )
        >>> me = MaskEstimator_v2.new({'combination': 'mul', 'ts_vad': 4, 'idim': 513})
        >>> me
        MaskEstimator_v2(
          combination='mul',
          (pre_net): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(513, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=513, bias=True)
            )
          )
          (post_net): Sequential(
            (birnn0): RNNP_packed(
              (net): ModuleList(
                (0): LSTM(513, 300, batch_first=True, bidirectional=True)
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
            (rearrange1): Rearrange('... spk time feature -> ... 1 time (spk feature)', spk=4)
            (birnn2): RNNP_packed(
              (net): ModuleList(
                (0): LSTM(1280, 300, batch_first=True, bidirectional=True)
                (1): Linear(in_features=600, out_features=320, bias=True)
              )
            )
            (linear2): Linear(in_features=320, out_features=2052, bias=True)
            (rearrange2): Rearrange('... 1 time (spk mask freq) -> ... spk mask time freq', mask=1, spk=4)
          )
          (final_activation): Sigmoid()
        )

        >>> frames, freq, num_speakers = 50, 513, 4
        >>> obs = torch.tensor(np.random.randn(frames, freq).astype(np.float32), requires_grad=True)
        >>> aux = [torch.tensor(np.random.randn(513).astype(np.float32), requires_grad=True) for _ in range(num_speakers)]
        >>> pprint(aux)
        [tensor(shape=(513,), dtype=float32),
         tensor(shape=(513,), dtype=float32),
         tensor(shape=(513,), dtype=float32),
         tensor(shape=(513,), dtype=float32)]
        >>> np.random.seed(0)
        >>> from padertorch.contrib.cb.track import track, tracker_list, ShapeTracker, ParameterTracker
        >>> with track(me, tracker_list(ShapeTracker, ParameterTracker)) as trackers:
        ...     _ = me(obs, aux)
        >>> print(trackers)
                                                  input                                       output                    #Params
          0 MaskEstimator_v2:    ([50, 513], [[513], [513], [513], [513]] ->                    ?                             0
                                                    )
          1   RNNP_packed:                     ([50, 513],)               ->                [50, 513]                         0
          2     LSTM:                          ([50, 513],)               ->    ([50, 600], ([2, 300], [2, 300]))     1_956_000
          3     Linear:                        ([50, 600],)               ->                [50, 513]                   308_313
          4   Sequential:                    ([4, 50, 513],)              ->             [4, 1, 50, 513]                      0
          5     RNNP_packed:                 ([4, 50, 513],)              ->               [4, 50, 320]                       0
          6       LSTM:                      ([4, 50, 513],)              -> ([4, 50, 600], ([2, 4, 300], [2, 4, 300] 1_956_000
                                                                                                ))
          7       Linear:                    ([4, 50, 600],)              ->               [4, 50, 320]                 192_320
          8     Dropout:                     ([4, 50, 320],)              ->               [4, 50, 320]                       0
          9     Tanh:                        ([4, 50, 320],)              ->               [4, 50, 320]                       0
         10     RNNP_packed:                 ([4, 50, 320],)              ->               [4, 50, 320]                       0
         11       LSTM:                      ([4, 50, 320],)              -> ([4, 50, 600], ([2, 4, 300], [2, 4, 300] 1_492_800
                                                                                                ))
         12       Linear:                    ([4, 50, 600],)              ->               [4, 50, 320]                 192_320
         13     Dropout:                     ([4, 50, 320],)              ->               [4, 50, 320]                       0
         14     Tanh:                        ([4, 50, 320],)              ->               [4, 50, 320]                       0
         15     Rearrange:                   ([4, 50, 320],)              ->              [1, 50, 1280]                       0
         16     RNNP_packed:                 ([1, 50, 1280],)             ->               [1, 50, 320]                       0
         17       LSTM:                      ([1, 50, 1280],)             -> ([1, 50, 600], ([2, 1, 300], [2, 1, 300] 3_796_800
                                                                                                ))
         18       Linear:                    ([1, 50, 600],)              ->               [1, 50, 320]                 192_320
         19     Linear:                      ([1, 50, 320],)              ->              [1, 50, 2052]                 658_692
         20     Rearrange:                   ([1, 50, 2052],)             ->             [4, 1, 50, 513]                      0
         21   Sigmoid:                      ([4, 1, 50, 513],)            ->             [4, 1, 50, 513]                      0


        """
        config["aux_net"] = None  # {'factory': AuxNet}
        if config["aux_net"] is None:
            # Assume I-Vectors by default
            config["aux_net_output_size"] = 100
        else:
            if issubclass(
                config["aux_net"]["factory"],
                (AuxNet),
            ):
                config["aux_net"]["idim"] = config["odim"] or config["idim"]

            if config["combination"] == "cat" and "odim" in config["aux_net"]:
                config["aux_net_output_size"] = config["aux_net"]["odim"]

    def __init__(
        self,
        *,
        idim=80,
        odim=None,
        layers=3,
        units=300,
        projs=320,
        dropout=0,
        nmask=1,
        pre_net="RNNP",
        aux_net: AuxNet,
        aux_net_output_size=None,
        combination: str = "cat",  # e.g. mul, cat
        ts_vad: [bool, int] = False,  # False or int (i.e. number of speakers)
        output_resolution: str = "tf",  # 'tf' or 't', t ... time/frame, f ... frequency
        random_speaker_order=True,
        num_averaged_permutations=1,  # In case of TS-VAD/SEP with last combination layer, try different permutations
        input_normalizer: InstanceNorm = None,
        aux_normalizer: InstanceNorm = None,
        explicit_vad=False,
    ):
        super().__init__()

        if odim is None:
            odim = idim
        self.odim = odim
        self.nmask = nmask
        self.output_resolution = output_resolution
        self.random_speaker_order = random_speaker_order
        self.num_averaged_permutations = num_averaged_permutations
        self.ts_vad = ts_vad
        self.input_normalizer = input_normalizer
        self.aux_normalizer = aux_normalizer
        self.explicit_vad = explicit_vad

        if not self.ts_vad:
            # num_averaged_permutations is only useful when using TS-VAD layer
            assert self.num_averaged_permutations == 1, (
                self.ts_vad,
                self.num_averaged_permutations,
            )

        if pre_net == "RNNP":
            pre_net = RNNP_packed(
                idim=idim,
                elayers=1,
                cdim=units,
                hdim=odim,
                dropout=dropout,
                typ="blstm",
            )

            self.pre_net = pre_net
        elif pre_net in [None, False]:
            self.pre_net = torch.nn.Identity()
        else:
            raise ValueError(pre_net)

        self.aux_net: AuxNet = aux_net
        self.combination = combination

        class SequentialDict:
            def __init__(self):
                self.data = collections.OrderedDict()
                self.idx = 0

            def __setitem__(self, key, value):
                for self.idx in range(self.idx, 100):
                    k = f"{key}{self.idx}"
                    if k not in self.data:
                        self.data[k] = value
                        return
                raise RuntimeError(key, value, self.data)

            def update(self, other):
                for k, v in other.items():
                    self[k] = v

        post_net = SequentialDict()

        ts_factor = 1

        if self.combination == "cat":
            if aux_net is None:
                assert aux_net_output_size is not None, (
                    self.combination,
                    aux_net_output_size,
                )
            else:
                assert aux_net_output_size == aux_net.odim, (
                    self.combination,
                    aux_net_output_size,
                    aux_net,
                )

            first_birnn_idim = odim + aux_net_output_size
        elif self.combination in ["mul", "film"]:
            first_birnn_idim = odim
        else:
            raise ValueError(self.combination)

        for l in range(layers):
            new_layers = collections.OrderedDict()

            if l == layers - 1 and ts_vad is not False:
                assert 2 < ts_vad < 20, ts_vad
                new_layers["rearrange"] = einops.layers.torch.Rearrange(
                    "... spk time feature -> ... 1 time (spk feature)",
                    spk=ts_vad,
                )
                ts_factor = ts_vad

            idim = (first_birnn_idim if l == 0 else projs) * ts_factor
            new_layers["birnn"] = RNNP_packed(
                idim=idim,
                elayers=1,
                cdim=units,  # * ts_factor,
                hdim=projs,  # * ts_factor,
                dropout=dropout,
                typ="blstm",
            )
            if l < layers - 1:
                new_layers["dropout"] = torch.nn.Dropout(p=dropout)
                new_layers["activation"] = torch.nn.Tanh()

            post_net.update(new_layers)

        if output_resolution in ["tf"]:
            final_out_features = (odim + explicit_vad) * nmask * ts_factor
            if ts_vad is False:
                final_einop = einops.layers.torch.Rearrange(
                    "... spk time (mask freq) -> ... spk mask time freq",
                    mask=self.nmask,
                )
            else:
                final_einop = einops.layers.torch.Rearrange(
                    "... 1 time (spk mask freq) -> ... spk mask time freq",
                    mask=self.nmask,
                    spk=ts_vad,
                )
        elif output_resolution == "t":
            assert explicit_vad is False, explicit_vad
            final_out_features = nmask * ts_factor
            if ts_vad is False:
                final_einop = einops.layers.torch.Reduce(
                    "... spk time mask -> ... spk mask time freq",
                    reduction="repeat",
                    mask=self.nmask,
                    freq=odim,
                )
            else:
                final_einop = einops.layers.torch.Reduce(
                    "... 1 time (spk mask) -> ... spk mask time freq",
                    reduction="repeat",
                    mask=self.nmask,
                    spk=ts_vad,
                    freq=odim,
                )
        else:
            raise ValueError(output_resolution)

        post_net["linear"] = torch.nn.Linear(
            in_features=projs,
            out_features=final_out_features,
        )
        post_net["rearrange"] = final_einop
        self.post_net = Sequential(post_net.data)
        self.final_activation = torch.nn.Sigmoid()

    def extra_repr(self) -> str:
        return f"combination={self.combination!r},"

    def forward(
        self,
        xs,
        aux=None,  # List of shape (time x freq)
    ) -> Output:
        """
        >>> from paderbox.utils.pretty import pprint
        >>> me = MaskEstimator_v2.new(dict(idim=257, ts_vad=3, output_resolution='tf'))
        >>> me = MaskEstimator_v2.new(dict(idim=257, ts_vad=False))
        >>> me
        MaskEstimator_v2(
          combination='cat',
          (pre_net): RNNP_packed(
            (net): ModuleList(
              (0): LSTM(257, 300, batch_first=True, bidirectional=True)
              (1): Linear(in_features=600, out_features=257, bias=True)
            )
          )
          (post_net): Sequential(
            (birnn0): RNNP_packed(
              (net): ModuleList(
                (0): LSTM(357, 300, batch_first=True, bidirectional=True)
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
            (linear2): Linear(in_features=320, out_features=257, bias=True)
            (rearrange2): Rearrange('... spk time (mask freq) -> ... spk mask time freq', mask=1)
          )
          (final_activation): Sigmoid()
        )
        >>> frames = 50
        >>> freq = 257
        >>> obs = torch.tensor(np.random.randn(frames, freq).astype(np.float32), requires_grad=True)
        >>> aux = [torch.tensor(np.random.randn(100).astype(np.float32), requires_grad=True) for _ in range(3)]
        >>> pprint(me(obs, aux))
        Output(mask=tensor(shape=(3, 1, 50, 257), dtype=float32),
               logit=tensor(shape=(3, 1, 50, 257), dtype=float32),
               embedding=tensor(shape=(3, 1, 100), dtype=float32),
               vad_mask=None,
               vad_logit=None)


        >>> me(obs, aux).mask.sum().backward()
        >>> for n, m in me.named_parameters():
        ...     print(n, torch.norm(m.grad))  # doctest: +ELLIPSIS
        pre_net.net.0.weight_ih_l0 tensor(...)
        pre_net.net.0.weight_hh_l0 tensor(...)
        pre_net.net.0.bias_ih_l0 tensor(...)
        pre_net.net.0.bias_hh_l0 tensor(...)
        pre_net.net.0.weight_ih_l0_reverse tensor(...)
        pre_net.net.0.weight_hh_l0_reverse tensor(...)
        pre_net.net.0.bias_ih_l0_reverse tensor(...)
        pre_net.net.0.bias_hh_l0_reverse tensor(...)
        pre_net.net.1.weight tensor(...)
        pre_net.net.1.bias tensor(...)
        post_net.birnn0.net.0.weight_ih_l0 tensor(...)
        post_net.birnn0.net.0.weight_hh_l0 tensor(...)
        post_net.birnn0.net.0.bias_ih_l0 tensor(...)
        post_net.birnn0.net.0.bias_hh_l0 tensor(...)
        post_net.birnn0.net.0.weight_ih_l0_reverse tensor(...)
        post_net.birnn0.net.0.weight_hh_l0_reverse tensor(...)
        post_net.birnn0.net.0.bias_ih_l0_reverse tensor(...)
        post_net.birnn0.net.0.bias_hh_l0_reverse tensor(...)
        post_net.birnn0.net.1.weight tensor(...)
        post_net.birnn0.net.1.bias tensor(...)
        post_net.birnn1.net.0.weight_ih_l0 tensor(...)
        post_net.birnn1.net.0.weight_hh_l0 tensor(...)
        post_net.birnn1.net.0.bias_ih_l0 tensor(...)
        post_net.birnn1.net.0.bias_hh_l0 tensor(...)
        post_net.birnn1.net.0.weight_ih_l0_reverse tensor(...)
        post_net.birnn1.net.0.weight_hh_l0_reverse tensor(...)
        post_net.birnn1.net.0.bias_ih_l0_reverse tensor(...)
        post_net.birnn1.net.0.bias_hh_l0_reverse tensor(...)
        post_net.birnn1.net.1.weight tensor(...)
        post_net.birnn1.net.1.bias tensor(...)
        post_net.birnn2.net.0.weight_ih_l0 tensor(...)
        post_net.birnn2.net.0.weight_hh_l0 tensor(...)
        post_net.birnn2.net.0.bias_ih_l0 tensor(...)
        post_net.birnn2.net.0.bias_hh_l0 tensor(...)
        post_net.birnn2.net.0.weight_ih_l0_reverse tensor(...)
        post_net.birnn2.net.0.weight_hh_l0_reverse tensor(...)
        post_net.birnn2.net.0.bias_ih_l0_reverse tensor(...)
        post_net.birnn2.net.0.bias_hh_l0_reverse tensor(...)
        post_net.birnn2.net.1.weight tensor(...)
        post_net.birnn2.net.1.bias tensor(...)
        post_net.linear2.weight tensor(...)
        post_net.linear2.bias tensor(...)

        >>> from padertorch.contrib.cb.track import track, tracker_list, ShapeTracker, ParameterTracker
        >>> with track(me, tracker_list(ShapeTracker, ParameterTracker)) as trackers:
        ...     _ = me(obs, aux)
        >>> print(trackers)
                                               input                                    output                    #Params
          0 MaskEstimator_v2:    ([50, 257], [[100], [100], [100]]) ->                    ?                             0
          1   RNNP_packed:                  ([50, 257],)            ->                [50, 257]                         0
          2     LSTM:                       ([50, 257],)            ->    ([50, 600], ([2, 300], [2, 300]))     1_341_600
          3     Linear:                     ([50, 600],)            ->                [50, 257]                   154_457
          4   Sequential:                 ([3, 50, 357],)           ->             [3, 1, 50, 257]                      0
          5     RNNP_packed:              ([3, 50, 357],)           ->               [3, 50, 320]                       0
          6       LSTM:                   ([3, 50, 357],)           -> ([3, 50, 600], ([2, 3, 300], [2, 3, 300] 1_581_600
                                                                                          ))
          7       Linear:                 ([3, 50, 600],)           ->               [3, 50, 320]                 192_320
          8     Dropout:                  ([3, 50, 320],)           ->               [3, 50, 320]                       0
          9     Tanh:                     ([3, 50, 320],)           ->               [3, 50, 320]                       0
         10     RNNP_packed:              ([3, 50, 320],)           ->               [3, 50, 320]                       0
         11       LSTM:                   ([3, 50, 320],)           -> ([3, 50, 600], ([2, 3, 300], [2, 3, 300] 1_492_800
                                                                                          ))
         12       Linear:                 ([3, 50, 600],)           ->               [3, 50, 320]                 192_320
         13     Dropout:                  ([3, 50, 320],)           ->               [3, 50, 320]                       0
         14     Tanh:                     ([3, 50, 320],)           ->               [3, 50, 320]                       0
         15     RNNP_packed:              ([3, 50, 320],)           ->               [3, 50, 320]                       0
         16       LSTM:                   ([3, 50, 320],)           -> ([3, 50, 600], ([2, 3, 300], [2, 3, 300] 1_492_800
                                                                                          ))
         17       Linear:                 ([3, 50, 600],)           ->               [3, 50, 320]                 192_320
         18     Linear:                   ([3, 50, 320],)           ->               [3, 50, 257]                  82_497
         19     Rearrange:                ([3, 50, 257],)           ->             [3, 1, 50, 257]                      0
         20   Sigmoid:                   ([3, 1, 50, 257],)         ->             [3, 1, 50, 257]                      0

        """
        if len(xs.shape) == 2:
            batched = False
            if self.random_speaker_order:
                perm = np.random.permutation(len(aux))
                iperm = np.argsort(perm)
                aux = [aux[i] for i in perm]

            if self.aux_net is not None:
                aux: torch.Tensor = self.aux_net(aux, xs)
            else:
                aux = torch.stack(aux, dim=0)
            spk = aux.shape[0]
        elif len(xs.shape) == 3:
            batched = True
            if self.random_speaker_order:
                perm = [
                    np.random.permutation(len(aux[0])) for _ in range(len(aux))
                ]
                iperm = np.argsort(perm, axis=-1)
                aux = [
                    [a[i] for i in perm_]
                    for a, perm_ in zip(aux, perm, strict=True)
                ]

            if self.aux_net is not None:
                assert self.aux_normalizer is None, (
                    self.aux_normalizer,
                    "Not clear, whether before or after",
                )
                aux: torch.Tensor = self.aux_net(aux, xs, batched=batched)
            else:
                if isinstance(aux, (tuple, list)):
                    aux = torch.stack(
                        [
                            (
                                torch.stack(a, dim=0)
                                if isinstance(a, (tuple, list))
                                else a
                            )
                            for a in aux
                        ],
                        dim=0,
                    )
                if self.aux_normalizer is not None:
                    aux = self.aux_normalizer(aux)
            spk = aux.shape[1]
        else:
            raise RuntimeError(xs.shape)

        if self.input_normalizer is not None:
            xs = self.input_normalizer(xs)
        xs: torch.Tensor = self.pre_net(xs)

        if len(aux.shape) == (3 if batched else 2):
            # Add a dummy dimension for I-Vectors and the sequence summary. Attention doesn't need this.
            aux = torch.unsqueeze(aux, dim=-2)
            aux_time = "1"
        elif len(aux.shape) == (4 if batched else 3):
            aux_time = "time"
        else:
            raise Exception(aux.shape, xs.shape)

        if self.combination == "mul":
            # learning hidden unit contributions (lhuc) (element-wise multiplication technique used in speaker beam)
            assert len(xs.shape) == (3 if batched else 2), xs.shape
            xs = xs[..., None, :, :] * aux
        elif self.combination == "film":
            # ToDO: Feature-wise Linear Modulation, e.g. for music https://arxiv.org/pdf/1907.01277.pdf (origin video)
            xs = xs * aux
            raise NotImplementedError(self.combination)
        elif self.combination == "cat":
            xs = torch.concat(
                [
                    einops.repeat(
                        xs, "... time feature -> ... spk time feature", spk=spk
                    ),
                    einops.repeat(
                        aux,
                        f"... spk {aux_time} feature -> ... spk time feature",
                        # 'spk 1 feature -> spk time feature',  # Sequence Summary and I-Vectors
                        # 'spk time feature -> spk time feature',  # Attention
                        time=xs.shape[-2],
                    ),
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(self.combination)

        # xs.shape: ... spk time feature

        if self.num_averaged_permutations == 1:
            pass
        else:
            if not batched:
                # Add batch dim
                xs = xs[None]

            assert (
                self.num_averaged_permutations > 1
            ), self.num_averaged_permutations
            trials = self.num_averaged_permutations

            speakers = xs.shape[-3]
            idx = (
                (np.arange(speakers)[:, None] + np.arange(speakers)[None, :])
                % speakers
            )[:trials, :].ravel()
            # e.g. idx == [0, 1, 2, 3, 4, 4, 0, 1, 2, 3] if speakers is 4 and trials is 2

            xs = einops.rearrange(
                xs[:, idx, :, :],
                "batch (trials speaker) time feat -> (batch trials) speaker time feat",
                trials=trials,
                speaker=speakers,
            )

        logit = self.post_net(xs)

        if self.num_averaged_permutations == 1:
            pass
        else:
            assert (
                self.num_averaged_permutations > 1
            ), self.num_averaged_permutations
            trials = self.num_averaged_permutations

            logit = einops.rearrange(
                logit,
                "(batch trials) speaker mask time feat -> batch (trials speaker) mask time feat ",
                trials=trials,
                speaker=speakers,
            )
            revert_idx = np.argsort(idx.ravel())
            # e.g. revert_idx == [0, 7, 1, 4, 2, 5, 3, 6]  if speakers is 4 and trials is 2
            logit = logit[:, revert_idx, :, :]
            logit = einops.rearrange(
                logit,
                # Note: Here speaker trials are swapped because of argsort
                "batch (speaker trials) mask time feat -> batch speaker trials mask time feat",
                trials=trials,
                speaker=speakers,
            ).mean(dim=-4)

            if not batched:
                # Remove batch dim
                logit = torch.squeeze(logit, dim=0)

        if self.random_speaker_order:
            # 3, 1, 50, 257
            # logit.shape: batch, speaker, mask, time, freq
            # logit = logit[..., iperm, :, :, :]
            ndim = len(logit.shape)
            if ndim == 4:
                logit = logit[iperm, :, :, :]
            elif ndim == 5:
                logit = logit[np.arange(len(logit))[:, None], iperm, :, :, :]
            else:
                raise ValueError(logit.shape)

        if self.explicit_vad:
            mask = self.final_activation(logit)
            vad_mask = mask[..., 0]
            mask = mask[..., 1:] * vad_mask[..., None]

            return Output(
                mask=mask,
                logit=None,
                vad_mask=vad_mask,
                vad_logit=logit[..., 0],
                embedding=aux,
            )
        else:
            return Output(
                mask=self.final_activation(logit),
                logit=logit,
                embedding=aux,
            )


def padded_sequence_reduction(
    xs: torch.Tensor, sequence_lengths, sequence_axis, batch_axis, op: str
):
    """
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> a, b = torch.tensor(np.random.randn(3, 4)), torch.tensor(np.random.randn(5, 4))
    >>> xs = torch.nn.utils.rnn.pad_sequence([a, b], batch_first=True)
    >>> xs.shape
    torch.Size([2, 5, 4])
    >>> xs
    tensor([[[ 1.7641,  0.4002,  0.9787,  2.2409],
             [ 1.8676, -0.9773,  0.9501, -0.1514],
             [-0.1032,  0.4106,  0.1440,  1.4543],
             [ 0.0000,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000]],
    <BLANKLINE>
            [[ 0.7610,  0.1217,  0.4439,  0.3337],
             [ 1.4941, -0.2052,  0.3131, -0.8541],
             [-2.5530,  0.6536,  0.8644, -0.7422],
             [ 2.2698, -1.4544,  0.0458, -0.1872],
             [ 1.5328,  1.4694,  0.1549,  0.3782]]], dtype=torch.float64)
    >>> padded_sequence_reduction(xs, [3, 5], 1, 0, 'mean')
    tensor([[ 1.1761, -0.0555,  0.6910,  1.1813],
            [ 0.7009,  0.1170,  0.3644, -0.2143]], dtype=torch.float64)
    >>> padded_sequence_reduction(xs, torch.tensor([3, 5]), 1, 0, 'mean')
    tensor([[ 1.1761, -0.0555,  0.6910,  1.1813],
            [ 0.7009,  0.1170,  0.3644, -0.2143]], dtype=torch.float64)
    >>> torch.stack([a.mean(0), b.mean(0)], 0)
    tensor([[ 1.1761, -0.0555,  0.6910,  1.1813],
            [ 0.7009,  0.1170,  0.3644, -0.2143]], dtype=torch.float64)

    """
    if sequence_lengths is None:
        if op == "sum":
            return torch.sum(xs, dim=sequence_axis)
        elif op == "mean":
            return torch.mean(xs, dim=sequence_axis)
        else:
            raise ValueError(op)

    if not torch.is_tensor(sequence_lengths):
        sequence_lengths = torch.Tensor(sequence_lengths).long().to(xs.device)

    mask = pt.ops.sequence.mask.compute_mask(
        xs,
        sequence_lengths=sequence_lengths,
        sequence_axis=sequence_axis,
        batch_axis=batch_axis,
    )
    if op == "sum":
        return torch.sum(xs * mask, dim=sequence_axis)
    elif op == "mean":
        shape = [1 for _ in xs.shape]
        shape[batch_axis] = xs.shape[batch_axis]
        del shape[sequence_axis]
        return torch.sum(
            xs * mask, dim=sequence_axis
        ) / sequence_lengths.reshape(shape)
    else:
        raise ValueError(op)
