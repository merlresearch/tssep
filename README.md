<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
## TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings

[![IEEE DOI](https://img.shields.io/badge/IEEE/DOI-10.1109/TASLP.2024.3350887-blue.svg)](https://doi.org/10.1109/TASLP.2024.3350887)
[![arXiv](https://img.shields.io/badge/arXiv-2303.03849-b31b1b.svg)](https://arxiv.org/abs/2303.03849)

This repository contains the core code that was used for the TS-VAD
and TS-SEP experiments in our 2024 IEEE/ACM TASLP article,
**TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings**
by Christoph Boeddeker, Aswin Shanmugam Subramanian, Gordon Wichern, Reinhold Haeb-Umbach, Jonathan Le Roux
([IEEE Xplore](https://doi.org/10.1109/TASLP.2024.3350887), [arXiv](https://arxiv.org/abs/2303.03849)).

If you use any part of this code for your work, we ask that you include the following citation:

    @article{Boeddeker2024feb,
    author = {Boeddeker, Christoph and Subramanian, Aswin Shanmugam and Wichern, Gordon and Haeb-Umbach, Reinhold and Le Roux, Jonathan},
    title = {{TS-SEP}: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings},
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    year = 2024,
    volume = 32,
    pages = {1185--1197},
    month = feb,
    }


# Installation

First install [pytorch](https://pytorch.org/get-started/locally/)
(torch, torchvision, and torchaudio) with a supported CUDA version for your
system.

Then install tssep:

```
git clone https://github.com/merlresearch/tssep.git
cd tssep
pip install -e .  # `pip install -e .[all]` to install test dependencies
```

# TS-VAD and TS-SEP experiment

This repository contains the core code that was used for the TS-VAD
and TS-SEP experiments in our publication.
Additionally, it contains a toy experiment that can be used to get started
([tssep/exp/run_tsvad.py](tssep/exp/run_tsvad.py) and [tssep/exp/run_tssep.py](tssep/exp/run_tssep.py)).

Before starting the training, set the following environment variables:
```
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0  # only necessary if you have more than one GPU
```

You can start the training with the following command:

```
python -m tssep.exp.run_tsvad
```

which will train a TS-VAD model on the toy data.
Next, you can train a TS-SEP model with the following command:

```
python -m tssep.exp.run_tssep
```

which will train a TS-SEP model on the toy data using the best checkpoint
from the TS-VAD model.

The experiments will create the folders `tssep/exp/tsvad` and `tssep/exp/tssep`,
where the checkpoints, logs, and configuration files are stored.
With [tensorboard](https://www.tensorflow.org/tensorboard), you can monitor the
training progress.

To run the model on the LibriCSS dataset, you have to replace the training
data with simulated LibriSpeech meetings.
Check [https://github.com/fgnt/tssep_data](https://github.com/fgnt/tssep_data)
for an example.


# How do I change something in an experiment?

To document the experiments, a `config.yaml` is written to the disk.
There you can check what the current parameters are.

Note: Check https://docs.google.com/presentation/d/1SKXlj34niGxVlcTnGAt4KTcymKaAfg7KCNYMO4C1Kho/edit#slide=id.g852ae286d5_3_40
or https://github.com/fgnt/padertorch/blob/master/doc/configurable.md if you want to know how to read a config.

To change a parameter, you can use the command line (e.g., `python -m tssep.train.run (init|train) with config.yaml my.parameter=abc`,
see [sacred CLI](https://sacred.readthedocs.io/en/stable/command_line.html)),
change/add a `named_config` [sacred CLI](https://sacred.readthedocs.io/en/stable/command_line.html) in the source code,
or change the "config.yaml" manually after the "init" command and before the "train" command.

If you have more advanced changes in mind, your can replace the `factory`s in the config
with your own classes.

# Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following file:
* `tssep/train/rnnp.py`

was adapted from https://github.com/espnet/espnet (license included in [LICENSES/Apache-2.0.md](LICENSES/Apache-2.0.md)):

```
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 ESPnet Developers
```


The following file:
* `tssep/train/feature_extractor_torchaudio.py`

was adapted from https://github.com/pytorch/audio (license included in [LICENSES/BSD-2-Clause.txt](LICENSES/BSD-2-Clause.txt)):

```
Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 torchaudio developers Developers
```
