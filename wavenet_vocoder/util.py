# coding: utf-8
from __future__ import with_statement, print_function, absolute_import


def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)

class Tee(object):
    """
    Emulates tee, copying sys.stdout to a file
    https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

def get_gpu_memory_map():
    """Get the current gpu usage.
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    # if CUDA_VISIBLE_DEVICES is not set to an integer then remove --id option
    exclude_id = os.environ['CUDA_VISIBLE_DEVICES'] in ['', 'NoDevFiles']
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
            '--format=csv,nounits,noheader', '--id={}'.format(os.environ['CUDA_VISIBLE_DEVICES'])
        ][:3 if exclude_id else None], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [list(map(int, x.split(', '))) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def print_gpu_memory():
    gpu_memory_map = get_gpu_memory_map()
    usage_str = '\n'.join(map(lambda kv: 'GPU-{} {}MB / {}MB ({}MB free)'.format(kv[0], *kv[1]), gpu_memory_map.items()))
    print('\n--- GPU MEMORY USAGE ---\n{}\n'.format(usage_str))
