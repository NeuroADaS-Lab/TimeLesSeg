import os
import zipfile
import requests
from tqdm import tqdm

# ALL THIS IS BORROWED and minimally adapted FROM github.com/MIC-DKFZ/HD-BET

def install_model_from_zip_file(zip_file: str, target_folder: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_folder)


def download_file(url: str, local_filename: str, chunk_size: int | None = 8192 * 16) -> str:
    # borrowed from https://github.com/MIC-DKFZ/HD-BET which in turn was:
    # # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(local_filename, 'wb'), "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename


def download_parameters(url: str, target_folder: str):
    fname = download_file(url, os.path.join(target_folder, 'tmp_trained_models.zip'))
    install_model_from_zip_file(fname, target_folder)
    os.remove(fname)


def recursively_remove_files_and_folders(folder_or_file: str):
    if os.path.isfile(folder_or_file):
        os.remove(folder_or_file)
        return

    maybe_files_or_folders = map(lambda x: os.path.join(folder_or_file, x), os.listdir(folder_or_file))
    for f in maybe_files_or_folders:
        recursively_remove_files_and_folders(f)

    os.rmdir(folder_or_file)
    return
