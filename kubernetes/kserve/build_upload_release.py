# Not sure I can assume ts is installed on the dev machine, alternative is reading ts/version.txt
from ts import version

# Move this to common utils function?
def try_and_handle(cmd, dry_run = False):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check = True)
        except subprocess.CalledProcessError as e:
            raise(e)

try_and_handle(f"./build_image.sh  -t pytorch/torchserve-kfs:{version()}")
try_and_handle(f"./build_image.sh -g -t pytorch/torchserve-kfs:{version()}-gpu")

for image in [f"pytorch/torchserve-kfs:{version()}", f"pytorch/torchserve-kfs:{version()}-gpu"]:
    try_and_handle(f"docker push {image}")
