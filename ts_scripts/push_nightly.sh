set -ex

rm -r dist || true


python setup.py --override-name torchserve-nightly bdist_wheel

# if [ -z "$PYPI_TOKEN" ]; then
#     echo "must specify PYPI_TOKEN"
#     exit 1
# fi

# python3 -m twine upload \
#     --username __token__ \
#     --password "$PYPI_TOKEN" \
#     dist/torchserve_nightly-*
