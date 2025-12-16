import time
from contextlib import contextmanager

@contextmanager
def time_block(label: str, logger):
    t0 = time.time()
    yield
    dt = time.time() - t0
    logger.info(f"{label} took {dt:.2f}s")
