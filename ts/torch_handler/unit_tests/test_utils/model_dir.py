import shutil
from pathlib import Path
from typing import Dict
from urllib import request
from urllib.parse import urlparse

REPO_DIR = Path(__file__).parents[4]


def download_model(model_url: str, model_dir: Path) -> None:
    cache_dir = REPO_DIR / ".cache"

    if not cache_dir.exists():
        cache_dir.mkdir()

    parts = urlparse(model_url)

    filename = Path(parts.path).name

    if not cache_dir.joinpath(filename).exists():
        with request.urlopen(model_url) as fin:
            with open(cache_dir / filename, "wb") as fout:
                fout.write(fin.read())

    shutil.copy(cache_dir / filename, model_dir / "model.pt")


def copy_files(src_dir: Path, dst_dir: Path, files: Dict) -> None:
    for src_file, dst_file in files.items():
        shutil.copyfile(src_dir / src_file, dst_dir / dst_file)
