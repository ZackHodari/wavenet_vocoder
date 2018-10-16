# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams


def preprocess(mod, in_dir, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    train_metadata = []
    valid_metadata = []
    test_metadata = []

    if hparams.valid_num_samples is None:
        valid_num_samples = int(hparams.valid_size * len(metadata))
    else:
        valid_num_samples = hparams.valid_num_samples

    if hparams.test_num_samples is None:
        test_num_samples = int(hparams.test_size * len(metadata))
    else:
        test_num_samples = hparams.test_num_samples

    if len(metadata[0]) == 5:
        # order by speaker_id
        metadata = sorted(metadata, key=lambda m: m[4])

        # separate by speaker_id
        speaker_ids = list(map(lambda m: m[4], metadata))
        num_speakers = max(speaker_ids) + 1
        for speaker_id in range(num_speakers):
            idx1 = speaker_ids.index(speaker_id)
            idx2 = idx1 + test_num_samples
            idx3 = idx1 + test_num_samples + valid_num_samples
            idx4 = speaker_ids.index(speaker_id + 1) if speaker_id + 1 < num_speakers else None

            test_metadata.extend(metadata[idx1:idx2])
            valid_metadata.extend(metadata[idx2:idx3])
            train_metadata.extend(metadata[idx3:idx4])
    else:
        idx1 = 0
        idx2 = test_num_samples
        idx3 = test_num_samples + valid_num_samples
        idx4 = None

        test_metadata.extend(metadata[idx1:idx2])
        valid_metadata.extend(metadata[idx2:idx3])
        train_metadata.extend(metadata[idx3:idx4])

    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train_metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    with open(os.path.join(out_dir, 'valid.txt'), 'w', encoding='utf-8') as f:
        for m in valid_metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    with open(os.path.join(out_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        for m in test_metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    print("Sampling frequency: {}".format(hparams.sample_rate))

    assert name in ["cmu_arctic", "ljspeech", "librivox", "jsut"]
    mod = importlib.import_module(name)
    preprocess(mod, in_dir, out_dir, num_workers)
