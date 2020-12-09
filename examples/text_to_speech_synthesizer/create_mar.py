import tempfile
import sys
sys.path.append('../../ts_scripts')
from ts_scripts.shell_utils import rm_dir, rm_file, download_save
import os
import shutil
import subprocess

TMP_DIR = tempfile.gettempdir()
rm_file("torchhub.zip")
rm_dir(os.path.join(TMP_DIR, "DeepLearningExamples-torchhub"))
rm_dir("PyTorch")
download_save("https://github.com/nvidia/DeepLearningExamples/archive/torchhub.zip", TMP_DIR)
shutil.unpack_archive(os.path.join(TMP_DIR, "torchhub.zip"), extract_dir=TMP_DIR)
rm_file("tacotron.zip")
shutil.copytree(os.path.join(TMP_DIR, "DeepLearningExamples-torchhub/PyTorch/SpeechSynthesis/"),
                "PyTorch/SpeechSynthesis/")
shutil.make_archive("tacotron", "zip", base_dir="PyTorch")
download_save("https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_fp32/versions/1/files/"
              "nvidia_tacotron2pyt_fp32_20190306.pth")
download_save("https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/"
              "nvidia_waveglowpyt_fp32_20190306.pth")
subprocess.run("torch-model-archiver --model-name waveglow_synthesizer --version 1.0 "
               "--model-file waveglow_model.py --serialized-file nvidia_waveglowpyt_fp32_20190306.pth "
               "--handler waveglow_handler.py "
               "--extra-files tacotron.zip,nvidia_tacotron2pyt_fp32_20190306.pth",
               check=True,
               shell=True)
rm_file("nvidia*", regex=True)
