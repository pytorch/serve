# Source: https://github.com/pytorch/torchx/blob/main/scripts/spellcheck.sh
set -ex
sudo apt-get install aspell
pyspelling -c ts_scripts/spellcheck_conf/spellcheck.yaml