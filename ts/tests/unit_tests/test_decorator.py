from ts.utils.serve_decorator import serve, torchserve_start, torchserve_stop, archive_model, create_handler, create_torchserve_config


@serve
def inference():
    return NotImplemented


if __name__ == "__main__":
    torchserve_start()

    torchserve_stop()
