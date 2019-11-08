


def handle(data, ctx):
    # Data is not none in prediction request
    # Python raises MemoryError when the python program goes out of memory. MMS expects this error from the handler
    # if the handlers can not allocate any further memory.
    if data is not None:
        raise MemoryError("Some Memory Error")
    return ["OK"]
