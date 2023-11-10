import multiprocessing
from typing import (
    List,
    Dict,
    TypeVar,
    Callable,
)
import time

def all_processes_dead(procs: List[multiprocessing.Process]) -> bool:
    for proc in procs:
        if proc.is_alive():
            return False
    return True

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def transform_keys(d: Dict[X, Z], fn: Callable[[X], Y]) -> Dict[Y, Z]:
    return {fn(key): value for key, value in d.items()}

def retry(
    fn: Callable,
    count: int,
    logging_fn: Callable,
    base_message: str,
    gap_seconds: float = 0.0,
):
    i = 0
    while i <= count:
        try:
            fn()
            return
        except Exception as exc:
            msg = base_message
            if i >= count:
                raise exc

            if i == 0:
                msg = f"{msg} Retrying..."
            else:
                msg = f"{msg} Retrying. Retry count: {i}"
            logging_fn(msg)
            i += 1
            time.sleep(gap_seconds)