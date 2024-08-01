# Copyright (c) 2024 Mitsubishi Electric Research Labs (MERL)
# Copyright (c) 2022 ESPnet developers (Apache License 2.0)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

import padertorch as pt
import torch
from einops import rearrange


class RNNP_packed(torch.nn.Module):
    """RNN with projection layer module

    Copied from ESPnet. Changes:
     - Use packed_sequence as input and don't convert it internally to a packed_sequence.
     - Remove subsample, I don't use it and it needs a lot of mem copies (packed to padded sequence and back)
     - Change code to produce a verbose repr (i.e. list all operations, e.g. last layer has no nonlinearity)
     - Disable dropout at eval time (see https://github.com/espnet/espnet/pull/1784#issuecomment-1110153016)

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type

    ToDo:
     - Fix for PackedSequence. TS-VAD and TS-SEP don't use it, because they
       are trained on fixed length data (chunk of a meeting).
    """

    def __init__(
        self,
        idim,
        elayers,
        cdim,
        hdim,
        dropout,
        typ="blstm",
        return_states=False,
    ):
        """

        >>> batch, frames, features = 2, 10, 512
        >>> rnn = RNNP_packed(512, 3, 300, 320, 0)
        >>> rnn
        RNNP_packed(
          (net): ModuleList(
            (0): LSTM(512, 300, batch_first=True, bidirectional=True)
            (1): Linear(in_features=600, out_features=320, bias=True)
            (2): Dropout(p=0, inplace=False)
            (3): Tanh()
            (4): LSTM(320, 300, batch_first=True, bidirectional=True)
            (5): Linear(in_features=600, out_features=320, bias=True)
            (6): Dropout(p=0, inplace=False)
            (7): Tanh()
            (8): LSTM(320, 300, batch_first=True, bidirectional=True)
            (9): Linear(in_features=600, out_features=320, bias=True)
          )
        )
        >>> rnn(torch.randn((batch, frames, features))).shape
        torch.Size([2, 10, 320])

        # >>> out = rnn(torch.nn.utils.rnn.pack_sequence([torch.randn((frames+5, features)), torch.randn((frames, features))]))
        # >>> out.data.shape
        # torch.Size([25, 320])
        # >>> out.batch_sizes
        # tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1])

        >>> speaker = 3
        >>> rnn(torch.randn((batch, speaker, frames, features))).shape
        torch.Size([2, 3, 10, 320])
        >>> rnn(torch.randn((frames, features))).shape
        torch.Size([10, 320])

        """
        super().__init__()
        bidir = typ[0] == "b"

        net = []

        for i in range(elayers):
            inputdim = idim if i == 0 else hdim

            RNN = torch.nn.LSTM if "lstm" in typ else torch.nn.GRU
            rnn = RNN(
                inputdim,
                cdim,
                num_layers=1,
                bidirectional=bidir,
                batch_first=True,
            )
            net.append(rnn)
            net.append(torch.nn.Linear(2 * cdim if bidir else cdim, hdim))

            if i < elayers - 1:
                net.append(torch.nn.Dropout(p=dropout))
                net.append(torch.nn.Tanh())

        self.net = torch.nn.ModuleList(net)

        self.elayers = elayers
        self.cdim = cdim
        self.typ = typ
        self.bidir = bidir
        self.dropout = dropout
        self.return_states = return_states

    def forward(self, xs_pack, prev_state=None):
        """RNNP forward

        Assume batch_first.

        :param torch.Tensor xs_pack: batch of packed input sequences (B, Tmax, idim)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        assert prev_state is None, prev_state
        # reset_backward_rnn_state(prev_state)

        if isinstance(xs_pack, torch.nn.utils.rnn.PackedSequence):
            batched = {2: False}[
                len(xs_pack.data.shape)
            ]  # Not implemented with multiple batch axis for packed sequence
        else:
            batched = {2: False, 3: False, 4: True}[len(xs_pack.shape)]

        if batched:
            h = rearrange(
                xs_pack, "batch speaker time freq -> (batch speaker) time freq"
            )
        else:
            h = xs_pack

        shape_batch = xs_pack.shape[0]
        del xs_pack

        elayer_states = []
        statefull_layer_idx = 0
        for layer in self.net:
            if isinstance(layer, torch.nn.RNNBase):
                h = [h]
                h, states = layer(
                    h.pop(),
                    hx=(
                        None
                        if prev_state is None
                        else prev_state[statefull_layer_idx]
                    ),
                )
                if self.return_states:
                    elayer_states.append(states)
                del states
                statefull_layer_idx += 1
            else:
                # sequence_elementwise works correctly as long the layer does not work on the batch or frame axis.
                h = [h]
                h = pt.ops.sequence.sequence_elementwise(layer, h.pop())

        if batched:
            h = rearrange(
                h,
                "(batch speaker) time freq -> batch speaker time freq",
                batch=shape_batch,
            )

        if self.return_states:
            return h, elayer_states
        else:
            return h
