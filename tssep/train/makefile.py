# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import padertorch as pt


def makefile(_config, eg, dump=True):
    storage_dir = Path(eg["trainer"]["storage_dir"])
    from padertorch.contrib.cb.io import SimpleMakefile

    main_python_path = pt.configurable.resolve_main_python_path()

    m = SimpleMakefile()
    m += "SHELL := /bin/bash"
    m.phony["help"] = "cat Makefile"
    m.phony["init"] = [
        "# Update config.yaml and Makefile. Print config.",
        f"python -m {main_python_path} init with config.yaml",
    ]
    m.phony["run"] = f"python -m {main_python_path} with config.yaml"

    m.phony["makefile"] = [
        "@# Update this makefile.",
        f"python -m {main_python_path} makefile with config.yaml",
    ]

    if dump:
        m.dump(storage_dir / "Makefile")
    return m
