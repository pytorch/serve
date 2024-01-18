from subprocess import Popen


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(
        command,
        shell=shell,
        close_fds=True,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=True,
    )
    if wait:
        cmd.wait()
    return cmd


def is_workflow(model_url):
    return model_url.endswith(".war")
