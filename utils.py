import GPUtil


def get_gpus(maxLoad=0.5, maxMemory=0.5):
    """Returns the available GPUs."""
    deviceIDs = GPUtil.getAvailable(
        order='first',
        limit=8,
        maxLoad=maxLoad,
        maxMemory=maxMemory,
        includeNan=False,
        excludeID=[],
        excludeUUID=[])
    return deviceIDs


def get_chunks(lst, n):
    """Returns list of n elements, constaining a sublist."""
    size = len(lst) // n
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])
    return chunks
