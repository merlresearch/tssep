# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import dataclasses

import lazy_dataset
import numpy as np


@dataclasses.dataclass
class DummyReader:
    # The actual data that is used for the training
    train_dataset_name: str = "train"

    # The data that is used to validate the model while it is training
    validate_dataset_name: str = "validate"

    # The data that is used to adapt the model to the target domain (i.e. eval data)
    domain_adaptation_src_dataset_name: str = "validate"

    # The data that is used to evaluate the model. The difference between eval
    # and validate is that eval uses a separate script and the evaluation is
    # too expensive to be done during training. While the validation writes
    # only logging information to the tensorboard tfevents file
    # (loss, spectrograms, etc.), the eval script writes the actual results
    # to the disk, that can be used for downstream applications, e.g., ASR.
    eval_dataset_name: str = "eval"

    sample_rate: int = 16000
    aux_size: int = 100
    train_examples: int = 10

    def _get_vad(self, num_samples, num_speakers):
        """
        >>> vad = DummyReader()._get_vad(71, 8)
        >>> for line in vad:
        ...     print(*['_#'[cell] for cell in line], sep='')
        ###############________________________________________________________
        ________###############________________________________________________
        ________________###############________________________________________
        ________________________###############________________________________
        ________________________________###############________________________
        ________________________________________###############________________
        ________________________________________________###############________
        ________________________________________________________###############
        >>> print(vad.sum(axis=1))
        [15 15 15 15 15 15 15 15]
        """
        vad = np.zeros((num_speakers, num_samples), dtype=bool)
        start = 0
        for i in range(num_speakers):
            end = num_samples * (i + 2) // (num_speakers + 1)
            vad[i, start:end] = True
            start = end - (end - start) // 2
        return vad

    def __call__(self, dataset_name, pre_load_apply, load_keys):
        num_speakers = 8
        num_channels = 1
        # num_samples = 16000 * 60  # The common length of the audio data is too long for the tests
        num_samples = self.sample_rate * 5

        if "train" in dataset_name:
            num_examples = self.train_examples
            start_seed = 0
        else:
            num_examples = 4
            start_seed = 0

        sinous = True

        # Fix the random state to allow overfitting on the dummy data.

        def get_example(seed):
            rng = np.random.RandomState(seed)
            if sinous:
                max_frequency = 7000
                min_frequency = 100
                num_frequencies = 3
                assert num_channels == 1, num_channels
                frequency = rng.randint(
                    min_frequency,
                    max_frequency,
                    size=(num_frequencies, num_speakers),
                )
                time = np.arange(num_samples) / self.sample_rate
                speaker_reverberation_early = (
                    np.sin(2 * np.pi * frequency[..., None] * time)
                    .sum(axis=0)
                    .astype(np.float32)
                )
                speaker_reverberation_early = speaker_reverberation_early[
                    :, None, :
                ]  # Add channel axis
            else:
                speaker_reverberation_early = rng.rand(
                    num_speakers, num_channels, num_samples
                ).astype(np.float32)
            vad = self._get_vad(num_samples, num_speakers)
            speaker_reverberation_early *= vad[:, None, :]

            noise = 1 * rng.rand(num_channels, num_samples).astype(np.float32)
            observation = speaker_reverberation_early.sum(axis=0) + noise

            if sinous:
                auxInput = np.full(
                    (num_speakers, self.aux_size),
                    fill_value=0,
                    dtype=np.float32,
                )
                scale = max_frequency + 1
                for spk, fs in enumerate(frequency.T):
                    for f in fs:
                        f = (f * auxInput.shape[1]) // scale
                        auxInput[spk, f : f + 2] = 1
            else:
                auxInput = rng.rand(num_speakers, self.aux_size).astype(
                    np.float32
                )

            r = {
                "example_id": f"dummy_id_{seed}",
                "num_samples": num_samples,
                "audio_data": {
                    "observation": observation,
                    "speaker_reverberation_early_ch0": speaker_reverberation_early[
                        :, 0
                    ],
                    "vad": vad,
                },
                "auxInput": auxInput,
                "dataset": dataset_name,
            }

            if "speaker_reverberation_early_ch0" not in load_keys:
                del r["audio_data"]["speaker_reverberation_early_ch0"]

            return r

        examples = [get_example(start_seed + i) for i in range(num_examples)]

        ds = lazy_dataset.new(examples)
        if pre_load_apply is not None:
            ds = pre_load_apply(ds)
        return ds

    # Unused here, but see tssep_data.data.data_hooks.ABC.pre_net for the motivation.
    class data_hooks:
        @staticmethod
        def pre_net(ex):
            return ex


def simple_toy_example(
    seed=0,
    frequency_bins=5,
):
    """
    Returns a simple toy example with partial overlap and Vad information,
    when the speakers are active.

    >>> from paderbox.utils.pretty import pprint
    >>> pprint(simple_toy_example())
    {'Observation': array(shape=(6, 79, 5), dtype=complex128),
     'Speech_reverberation_early': array(shape=(2, 6, 79, 5), dtype=complex128),
     'Vad': [ArrayInterval("0:55", shape=(79,)),
      ArrayInterval("45:79", shape=(79,))],
     'mask': array(shape=(3, 79, 5), dtype=float64)}
    """
    import einops
    from paderbox.array.interval import ArrayInterval
    from pb_bss.distribution.complex_angular_central_gaussian import (
        sample_complex_angular_central_gaussian,
    )

    rng = np.random.RandomState(seed)

    num_channels = 6
    time_frames = 79

    # Generate two speakers with different directions of arrival
    doa1 = np.exp(1j * np.array([0, 0, 0, 0, 0, 0][:num_channels]))
    doa2 = np.exp(
        1j * np.pi * np.array([0, 1, 0.5, 0.25, 0.75, 0][:num_channels])
    )
    cov1 = doa1[:, None] * doa1[None, :].conj() + 0.01 * np.eye(num_channels)
    cov2 = doa2[:, None] * doa2[None, :].conj() + 0.01 * np.eye(num_channels)

    np.random.seed(seed + 1)
    s1 = sample_complex_angular_central_gaussian(
        size=(time_frames * frequency_bins,), covariance=cov1
    )
    np.random.seed(seed + 2)
    s2 = sample_complex_angular_central_gaussian(
        size=(time_frames * frequency_bins,), covariance=cov2
    )

    s1 = einops.rearrange(
        s1,
        "(time frequency) ... -> ... time frequency",
        frequency=frequency_bins,
        time=time_frames,
    )
    s2 = einops.rearrange(
        s2,
        "(time frequency) ... -> ... time frequency",
        frequency=frequency_bins,
        time=time_frames,
    )

    dia = [
        ArrayInterval.from_str("0:55", shape=time_frames),
        ArrayInterval.from_str("45:79", shape=time_frames),
    ]
    for i, s in enumerate([s1, s2]):
        s[..., ~dia[i], :] = 0

    noise = 0.01 * rng.randn(num_channels, time_frames, frequency_bins)
    observation = s1 + s2 + noise

    from pb_bss.extraction.mask_module import wiener_like_mask

    mask = wiener_like_mask(np.array([s1, s2, noise]), sensor_axis=1)

    return {
        "Observation": observation,
        "Speech_reverberation_early": np.array([s1, s2]),
        "Vad": dia,
        "mask": mask,
    }
