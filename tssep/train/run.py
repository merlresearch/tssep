# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""

# How to train a model

Start the training by running the following command in the terminal:

    python -m tssep.train.run with config1.yaml config2.yaml ...

where config[1,2].yaml are the config file that you want to use.
The last config file has the highest priority.
For example config files, see the yaml files

    tssep/tssep/exp/[init_cfg_common.yaml,init_cfg_tsvad.yaml,init_cfg_tssep.yaml].

You may want to check

        tssep/tssep/exp/[run_tsvad.py,run_tssep.py]

for a call example, where the storage_dir is set.
For a productive example, check https://github.com/fgnt/tssep_data.
It replaces the toy data with the speech data (i.e. you have to replace the input data).

# Hints for customization

This run script use sacred to manage the configuration.
See https://sacred.readthedocs.io/ for the documentation.
To change a value in the config, you have multiple options:
 - Change the value in the config.yaml file that is generated in the storage
   dir.
 - Use the commandline interface, e.g.
   `python -m tssep.train.run with eg.trainer.storage_dir=/path/to/dir`
   to change the storage dir.

Compare configs:
    icdiff ../5/config.yaml config.yaml

"""

import os
import shlex
import sys

import psutil

if __name__ == "__main__":
    print(sys.argv)
    print(shlex.join(sys.argv))

import filecmp

import sacred.commands

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False

sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.extend(
    [
        k
        for k in os.environ.keys()
        if "cuda" in k.lower()
        or "slurm" in k.lower()
        or "srun" in k.lower()
        or k in ["OMP_NUM_THREADS"]
    ]
)

import datetime  # noqa
import shutil  # noqa
from pathlib import Path  # noqa

import paderbox as pb  # noqa
import padertorch as pt  # noqa
from sacred.commands import print_config  # noqa

from tssep.train.experiment import Experiment  # noqa

# Use the following line to print the config once it is published
# from padertorch.contrib.cb.sacred_helper import print_config


ex = sacred.Experiment("extract")


@ex.config
def config():
    script_python_path = (  # noqa
        pt.configurable.resolve_main_python_path()
    )  # Info inside the config to find which script produces the config

    eg = {"trainer": {"storage_dir": None}}
    Experiment.get_config(eg)  # Fill defaults

    if eg["trainer"]["storage_dir"] is None:
        eg["trainer"]["storage_dir"] = pt.io.get_new_subdir(
            Path(__file__).parent.parent.parent / "egs" / ex.path,
            consider_mpi=False,
            mkdir=False,
        )


def backup_config(config_yaml):
    time = datetime.datetime.fromtimestamp(
        os.stat(config_yaml).st_mtime
    ).strftime("%Y_%m_%d_%H_%M_%S")
    backup_file = (
        config_yaml.parent
        / "backup"
        / (config_yaml.name.replace(".yaml", f"_{time}.yaml"))
    )

    if backup_file.exists():
        if filecmp.cmp(backup_file, config_yaml):
            print(
                "Skip backup of config.yaml, because it already has a backup."
            )
            return
        else:
            print(
                'ERROR: Backup of config.yaml was changed. Use "now" as timestamp instead of modified time.'
            )
            time = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            backup_file = (
                config_yaml.parent
                / "backup"
                / (config_yaml.name.replace(".yaml", f"_{time}.yaml"))
            )
            assert not backup_file.exists(), backup_file
    else:
        backup_file.parent.mkdir(exist_ok=True)

    print(f"Create backup of config.yaml: {backup_file}")
    shutil.copy(config_yaml, backup_file)


def dump_config(storage_dir, _config):
    storage_dir = Path(storage_dir)

    config_yaml = Path(storage_dir / "config.yaml")
    config_yaml_content = pt.io.dumps_config(_config, ".yaml")

    if config_yaml.exists():
        backup_config(config_yaml)
        old_config_yaml_content = config_yaml.read_text()

        if config_yaml_content == old_config_yaml_content:
            print("Skip dump_config, because config has not changed.")

    pb.io.atomic.write_text_atomic(config_yaml_content, config_yaml)


@ex.command
def init(_run, _config, eg):
    storage_dir = Path(eg["trainer"]["storage_dir"])
    storage_dir.mkdir(exist_ok=True, parents=True)

    with open(storage_dir / "python_history.txt", "a") as fd:
        print(
            f"{shlex.join(psutil.Process().cmdline())}"
            f'  # {datetime.datetime.today().strftime("%Y.%m.%d %H:%M:%S")}'
            f"  # {Path.cwd()}",
            file=fd,
        )

    # Try to predict if the storage_dir is wrong in the config.
    #     When you are in <storage_root>/<id1>, but storage_dir is <storage_root>/<id2>,
    #     it is very likely that the storage_dir is wrong.
    cwd = Path.cwd()
    if cwd.parts[:-1] == storage_dir.parts[:-1]:
        assert cwd == storage_dir, (cwd, storage_dir)

    dump_config(storage_dir=storage_dir, _config=_config)

    ex.commands["print_config"]()
    ex.commands["makefile"]()

    eg: Experiment = Experiment.from_config(eg)
    eg.add_log_files()

    print(f"Initialized {storage_dir}")


@ex.main
def train(_run, _config, eg):
    ex.commands["init"]()
    eg: Experiment = Experiment.from_config(eg)
    eg.train()


from tssep.train.makefile import makefile  # noqa

makefile = ex.command(unobserved=True)(makefile)


if __name__ == "__main__":
    print(shlex.join(psutil.Process().cmdline()))

    # To debug, use --pdb commandline option from sacred
    ex.run_commandline()
