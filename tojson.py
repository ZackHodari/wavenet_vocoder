# coding: utf-8
"""
Dump hyper parameters to json file.

usage: tojson.py [options] <output_json_path>

options:
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext
import json

from hparams import hparams, hparams_debug_string

if __name__ == "__main__":
    args = docopt(__doc__)
    preset = args["--preset"]
    output_json_path = args["<output_json_path>"]

    os.makedirs(dirname(output_json_path), exist_ok=True)

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    j = hparams.values()

    # for compat legacy
    for k in ["preset", "presets"]:
        if k in j:
            del j[k]

    with open(output_json_path, "w") as f:
        json.dump(j, f, indent=2)
    sys.exit(0)
