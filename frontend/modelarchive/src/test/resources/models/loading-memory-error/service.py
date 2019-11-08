


def handle(ctx, data):
    # Python raises MemoryError when the python program goes out of memory. MMS expects this error from the handler
    # if the handlers can not allocate any further memory.
    raise MemoryError("Throwing memory error")
