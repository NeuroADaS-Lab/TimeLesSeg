import os
from typing import Callable, Union, Any, Iterable, List
import time
import logging
import json
import pickle
import re
import multiprocessing
import yaml

### Keep everything python==3.8 compatible here, please

# path-related utils
join = os.path.join
basename = os.path.basename
dirname = os.path.dirname
isdir = os.path.isdir
isfile = os.path.isfile
exists = os.path.exists
splitext = os.path.splitext
split = os.path.split
abspath = os.path.abspath

remove = os.remove
rmdir = os.rmdir
listdir = os.listdir

DEFAULT_NUM_PROCESSES = 12
def get_default_num_processes() -> int:
    return min(DEFAULT_NUM_PROCESSES, os.cpu_count())

def get_default_device() -> str:
    import torch
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        mps_available = getattr(torch.mps, 'is_available', getattr(torch.backends.mps, 'is_available', None))
        if mps_available is not None and mps_available():
            device = 'mps'
    return device

def maybe_mkdir(dir: str):
    os.makedirs(dir, exist_ok=True)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def load_yaml(yaml_file: str) -> dict:
    with open(yaml_file, mode='r') as f:
        content = yaml.safe_load(f)
    return content

def remove_nii_extension(nii_file: str) -> str:
    extension_start = nii_file.find('.nii')
    if extension_start == -1:
        return nii_file
    return nii_file[:extension_start]

def get_properties_file_from_nii(nii_file: str) -> str:
    return find_prediction_companion_from_nii(nii_file, suffix_target='.pkl')

def get_probabilities_file_from_nii(nii_file: str) -> str:
    return find_prediction_companion_from_nii(nii_file, suffix_target='.npz')

def find_prediction_companion_from_nii(nii_file: str, suffix_target: str) -> str:
    assert suffix_target in ['.pkl', '.npz']
    last_extension_index = nii_file.rfind('.nii')
    assert last_extension_index >= 0
    properties_file = nii_file[:last_extension_index] + suffix_target
    return properties_file

def check_input_lists(*lists: list, f: Callable) -> bool:
    """
    We check that:
    - 1: all lists have the same length
    - 2: f(x) holds for every element in all lists elementwise
    """
    l, *lists = lists
    return all((len(l) == len(ll) and all(f(l[i]) == f(ll[i]) for i in range(len(l)))) for ll in lists)

def get_image_file_ending(file: str) -> str:
    PATTERN = r'(.(?:nii(?:.gz)?|mha))'
    result = re.search(PATTERN, file)
    if result is None:
        raise ValueError('Invalid filename! Expected it to end in either .nii(.gz) or .mha! Got "%s"' % file)
    return result.group()

def timestampify(root: str = "") -> str:
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    if len(root) and not root.endswith('_'):
        root += '_'
    return root + timestamp

def initialize_tmp_path(root: str) -> str:
    tmp_path = timestampify(root)
    maybe_mkdir(tmp_path)
    return tmp_path

class TmpDir:
    """
    Use as follows:

    ```python
    with TmpDir(root) as tmpdir:
        do_x()
    ```
    """

    def __init__(self, root: str = None):
        if root is None:
            # we assume utils is directly under project folder
            root = dirname(__file__)
        self.tmpdir = initialize_tmp_path(root)

    def __enter__(self):
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        self.cleanup()

    def cleanup(self):
        files = [join(self.tmpdir, x) for x in listdir(self.tmpdir) if isfile(join(self.tmpdir, x))]
        for file in files:
            remove(file)
        rmdir(self.tmpdir)

# logging
def parse_logging_level(logging_level: str) -> int:
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/658264383bc1391443dcab8b2edd9af503df1e41/spinalcordtoolbox/utils/sys.py#L102
    assert logging_level.upper() in ['WARNING', 'ERROR', 'DEBUG', 'INFO', 'CRITICAL']
    return getattr(logging, logging_level)

def setup_loggers(*logger_names: str, verbosity: str, log_file: str, console_verbosity: str = None, return_logger: bool = False) -> Union[logging.Logger, None]:
    console_verbosity = console_verbosity or verbosity

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    fh.setLevel(verbosity)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(console_verbosity)

    for logger in logger_names:
        l = logging.getLogger(logger)

        # to avoid having a thousand fucking handlers
        if l.hasHandlers():
            continue

        l.setLevel(verbosity)
        l.addHandler(fh)
        l.addHandler(ch)

    # ugly as fuck
    if return_logger: return l


class DtoAttr:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._initializer_kwargs = kwargs

    def __getitem__(self, item):
        return getattr(self, item)


def _unpack_and_call(packet):
    """
    Unpacks a tuple of (function, args) and calls function(*args).
    """
    func, args = packet
    return func(*args)

def run_parallel(
    func: Callable,
    args_list: Iterable[tuple],
    num_processes: int,
    timeout: float = 2.0,
    preserve_order: bool = False
) -> List[Any]:
    """
    Runs `func` in parallel using the heartbeat pattern to detect crashes.
    
    :param func: The function to run (must be picklable).
    :param args_list: A list (or iterable) of tuples, where each tuple contains arguments for `func`.
                      e.g. [(img1, seg1), (img2, seg2), ...]
    :param num_processes: Number of workers.
    :param timeout: Time to wait for a result before checking for zombie workers.
    :param preserve_order: determines whether imap or imap_unordered is used.
    :return: List of results.
    """

    args_list = list(args_list)
    tasks = ((func, args) for args in args_list)
    total_tasks = len(args_list)

    results = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        # We map the generic '_unpack_and_call' function over our packed tasks
        multiprocessing_func = p.imap_unordered if not preserve_order else p.imap
        task_iterator = multiprocessing_func(_unpack_and_call, tasks)

        while len(results) < total_tasks:
            try:
                # Wait for the next result with a timeout
                res = task_iterator.next(timeout)
                results.append(res)

            except multiprocessing.TimeoutError:

                if all(not bool(w.exitcode) for w in p._pool):
                    continue
                # Find out which one died if possible, but usually just raising generic error is enough
                raise RuntimeError(f"A background worker crashed (likely OOM). Reduce num_processes.")

            except StopIteration:
                break

    return results

if __name__ == "__main__":
    lists = [
        ['y1', 'y2'],
        ['x1', 'x2'],
        ['z1', 'z2']
    ]

    assert check_input_lists(*lists, f=lambda x: x[-1])

    x = DtoAttr(something = 3, other = 4)
    assert x.something == x['something'] and hasattr(x, 'other') and x._initializer_kwargs == {'something': 3, 'other': 4}

    some_file = '/vicentcaselles/cooldataset/file.nii.gz'
    assert get_image_file_ending(some_file) == '.nii.gz', get_image_file_ending(some_file)

    some_other_file = '/vicentcaselles/cooldataset/other_file.nii'
    assert get_image_file_ending(some_other_file) == '.nii'

    final_file = '/vicentcaselles/cooldataset/labels/other_file.mha'
    assert get_image_file_ending(final_file) == '.mha', get_image_file_ending(final_file)


    some_other_file = '/vicentcaselles/cooldataset/other_file.nii'
    some_other_file_compressed = '/vicentcaselles/cooldataset/other_file.nii.gz'
    x = remove_nii_extension(some_other_file)
    y = remove_nii_extension(some_other_file_compressed)
    assert x == y


    pdf_file = '/pdiddy/personal/baby_oil_bills.pdf'
    assert remove_nii_extension(pdf_file) == pdf_file
