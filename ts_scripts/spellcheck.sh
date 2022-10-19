# Source: https://github.com/pytorch/torchx/blob/main/scripts/spellcheck.sh
set -ex
sudo apt-get install aspell

if [[ -z "$@" ]]; then
    sources=$(find -name '*.md')
else
    sources=$@
fi

sources_arg=""
for src in $sources ;do
        sources_arg+=" -S $src"
done
pyspelling -c ts_scripts/spellcheck_conf/spellcheck.yaml --name Markdown $sources_arg