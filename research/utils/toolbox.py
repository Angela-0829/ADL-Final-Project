'''Store all useful functions.'''
import sys
import time
import random
import torch
import numpy as np
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, \
    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def same_seed(seed: int):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_controller(func):
    '''Busy waiting when no gpu avaliable.'''
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print('SigInt trigger...')
                sys.exit()
            except RuntimeError:
                print('[Error] CUDA Memory Insufficient, retry after 30 secondes.')
                time.sleep(30)

    return wrapper


def get_free_gpu() -> str:
    '''Return gpu device which has more avaliable memory.'''
    nvmlInit()
    memory = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        memory.append(meminfo.free)

    nvmlShutdown()
    return f"cuda:{np.argmax(memory)}"
