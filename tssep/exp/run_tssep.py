# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shlex
import sys
from pathlib import Path


class c:
    green = "\033[92m"
    red = "\033[91m"
    purple = "\033[95m"
    cyan = "\033[96m"
    end = "\033[0m"


def run(cmd, failure):
    cmd = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f"{c.green}$ {cmd}{c.end}")

    # os.system handles a KeyboardInterrupt better than subprocess.run
    returncode = os.system(cmd)

    if returncode != 0:
        print(
            f"{c.red}$ {cmd}{c.end} failed with {c.red}return code {returncode}{c.end}"
        )
        if failure == "exit":
            sys.exit(returncode)
        elif failure == "raise":
            raise RuntimeError(
                f"Command {cmd} failed with return code {returncode}"
            )
        else:
            raise ValueError(f"Unknown failure mode {failure}")


_cwd = Path(__file__).parent


def main(
    configs=(
        f"{_cwd}/init_cfg_common.yaml",
        f"{_cwd}/init_cfg_tssep.yaml",
    ),
    storage_dir=f"{_cwd}/tsvad",
    checkpoint=f"{_cwd}/tsvad/checkpoints/ckpt_best_loss.pth",
    failure="raise",
):
    storage_dir = Path(storage_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    configs = [os.fspath(Path(config).resolve()) for config in configs]

    cmd = [
        sys.executable,
        "-m",
        "tssep.train.run",
        "init",
        "with",
        *configs,
        f"eg.trainer.storage_dir={storage_dir}",
        f"eg.init_ckpt.init_ckpt={checkpoint}",
    ]
    if (storage_dir / "config.yaml").exists():
        print(
            f"{c.cyan}SEP storage dir {storage_dir} already exists. Skipping init.{c.end}"
        )
    else:
        run(cmd, failure=failure)

    cmd = f"{sys.executable} -m tssep.train.run with config.yaml"
    run(f"cd {storage_dir} && {cmd}", failure=failure)


if __name__ == "__main__":
    main(failure="exit")
