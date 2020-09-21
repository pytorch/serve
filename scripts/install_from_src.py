import platform
import install_utils as utils
import click


@click.command()
@click.option('--is_gpu', '-g', default=False, help='To be executed on GPU or CPU')
@click.option('--cuda_version', '-c', default='cuda102', help='Cuda version to be used')
def install_from_src(is_gpu, cuda_version):
    utils.clean_slate()
    if platform.system() == 'Windows':
        utils.install_torch_deps(is_gpu, cuda_version)
    else:
        utils.install_torch_deps_linux(is_gpu, cuda_version)
    utils.build_install_server()
    utils.build_install_archiver()
    utils.clean_up_build_residuals()


if __name__ == '__main__':
    install_from_src()
