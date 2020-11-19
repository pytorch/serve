import requests
import os
import shutil
import glob
from pathlib import Path


def download_save(url, path=None, filename=None):
    if not filename:
        filename = url.split('/')[-1]
    if path:
        filename = os.path.join(path, filename)
    print(f"Downloading from url : {url}")
    resp = requests.get(url)
    with open(filename, 'wb') as f:
        print(f"Saving data to file : {filename}")
        f.write(resp.content)


def rm_file(path, regex=False):
    if regex:
        file_list = glob.glob(path, recursive=True)
    else:
        file_list = [path]
    for file in file_list:
        path = Path(file)
        if os.path.exists(path):
            print(f"Removing file : {path}")
            os.remove(path)


def rm_dir(path):
    path = Path(path)
    if os.path.exists(path):
        print(f"Deleting directory : {path}")
        shutil.rmtree(path)


def unzip(filename, destination, arc_type):
    shutil.unpack_archive(filename, destination, arc_type)
    print("Archive file unpacked successfully.")
