import sys
import os
import logging
import importlib
import inspect
from enum import Enum
from threading import Lock, Thread
from anyio import Semaphore, to_thread
import time
from pathlib import Path

from typing import (
    Any, 
    Generator,
    AsyncGenerator, 
    Dict, 
    Optional, 
    Set, 
    Union
)

from utils.common import retry
from fastapi import HTTPException
from contextlib import asynccontextmanager

import asyncio

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from inference.model import Model as ModelClass

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES", "1"))
STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS = 60
DEFAULT_GENERATE_CONCURRENCY = 1

class DeferredSemaphoreManager:
    """
    Helper class for supported deferred semaphore release.
    """

    def __init__(self, semaphore: Semaphore):
        self.semaphore = semaphore
        self.deferred = False

    def defer(self):
        """
        Track that this semaphore is to be deferred, and return
        a release method that the context block can use to release
        the semaphore.
        """
        self.deferred = True

        return self.semaphore.release


@asynccontextmanager
async def deferred_semaphore(semaphore: Semaphore):
    """
    Context manager that allows deferring the release of a semaphore.
    It yields a DeferredSemaphoreManager -- in your use of this context manager,
    if you call DeferredSemaphoreManager.defer(), you will get back a function that releases
    the semaphore that you must call.
    """
    semaphore_manager = DeferredSemaphoreManager(semaphore)
    await semaphore.acquire()

    try:
        yield semaphore_manager
    finally:
        if not semaphore_manager.deferred:
            semaphore.release()


class ModelHandler:
    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3
    
    def __init__(
            self, 
            config: Dict
        ) -> None:
        self._config = config
        self._logger = logging.getLogger()
        self.name = MODEL_BASENAME
        self.ready = False
        self._load_lock = Lock()
        self._status = ModelHandler.Status.NOT_READY
        self._predict_semaphore = Semaphore(
            self._config.get("runtime", {}).get(
                "predict_concurrency", DEFAULT_GENERATE_CONCURRENCY
            )
        )
        self._background_tasks: Set[asyncio.Task] = set()

    def load(self) -> bool:
        """Responsible for loading the model. 
        It uses a lock (_load_lock) to ensure that only one loading process can occur at a time.

        Returns:
            bool: Model status
        """
        if self.ready:
            return self.ready

        # If the lock cannot be acquired, loading thread is already running; 
        if not self._load_lock.acquire(blocking=False):
            return False

        self._status = ModelHandler.Status.LOADING

        self._logger.info("Executing model.load()...")

        # Attempt to load the model
        try:
            start_time = time.perf_counter()
            self.try_load()
            self.ready = True
            self._status = ModelHandler.Status.READY
            self._logger.info(
                f"Completed model.load() execution in {_elapsed_ms(start_time)} ms"
            )

            return self.ready
        except Exception:
            self._logger.exception("Exception while loading model")
            self._status = ModelHandler.Status.FAILED
        finally:
            self._load_lock.release()

        return self.ready
        
    def start_load(self):
        """Initiates the load() method in a seaparate thread
        """
        if self.should_load():
            thread = Thread(target=self.load)
            thread.start()

    def load_failed(self) -> bool:
        return self._status == ModelHandler.Status.FAILED

    def should_load(self) -> bool:
        """Check if not loading and don't retry failed loads
        """
        return (
            not self._load_lock.locked()
            and not self._status == ModelHandler.Status.FAILED
            and not self.ready
        )
    
    def try_load(self):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Sanity check for ModelClass
        model_init_params = {}
        if _signature_accepts_keyword_arg(inspect.signature(ModelClass), "config"):
            model_init_params["config"] = self._config
        if _signature_accepts_keyword_arg(inspect.signature(ModelClass), "data_dir"):
            model_init_params["data_dir"] = data_dir

        # initialize th model class
        self._model = ModelClass(**model_init_params)

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warn,
                "Failed to load model.",
                gap_seconds=1.0,
            )
    
    async def generate(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        # It's possible for the user's predict function to be a:
        #   1. Generator function (function that returns a generator)
        #   2. Async generator (function that returns async generator)
        # In these cases, just return the generator or async generator,
        # as we will be propagating these up. No need for await at this point.
        #   3. Coroutine -- in this case, await the predict function as it is async
        #   4. Normal function -- in this case, offload to a separate thread to prevent
        #      blocking the main event loop
        if inspect.isasyncgenfunction(
            self._model.predict
        ) or inspect.isgeneratorfunction(self._model.predict):
            return self._model.predict(payload)

        if inspect.iscoroutinefunction(self._model.predict):
            return await _intercept_exceptions_async(self._model.predict)(payload)

        return await to_thread.run_sync(
            _intercept_exceptions_sync(self._model.predict), payload
        )
    
    async def __call__(
        self, body: Any, headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict, Generator]:
        """Main method to run model inference with the given input.

        Args:
            body (Any): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
            Generator: In case of streaming response
        """

        payload = body

        async with deferred_semaphore(self._predict_semaphore) as semaphore_manager:
            response = await self.generate(payload, headers)

            # Streaming cases
            if inspect.isgenerator(response) or inspect.isasyncgen(response):
                
                async_generator = _force_async_generator(response)

                if headers and headers.get("accept") == "application/json":
                    # In the case of a streaming response, consume stream
                    # if the http accept header is set, and json is requested.
                    return await _convert_streamed_response_to_string(async_generator)

                # To ensure that a partial read from a client does not cause the semaphore
                # to stay claimed, we immediately write all of the data from the stream to a
                # queue. We then return a new generator that reads from the queue, and then
                # exit the semaphore block.
                response_queue = asyncio.Queue()

                # This task will be triggered and run in the background.
                task = asyncio.create_task(
                    self.write_response_to_queue(response_queue, async_generator)
                )

                # We add the task to the ModelHandler instance to ensure it does
                # not get garbage collected after the predict method completes,
                # and continues running.
                self._background_tasks.add(task)

                # Defer the release of the semaphore until the write_response_to_queue
                # task.
                semaphore_release_function = semaphore_manager.defer()
                task.add_done_callback(lambda _: semaphore_release_function())
                task.add_done_callback(self._background_tasks.discard)

                async def _response_generator():
                    while True:
                        chunk = await asyncio.wait_for(
                            response_queue.get(),
                            timeout=STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS,
                        )
                        if chunk is None:
                            return
                        yield chunk.value

                return _response_generator()

        return response

def _elapsed_ms(since_micro_seconds: float) -> int:
    return int((time.perf_counter() - since_micro_seconds) * 1000)

def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)

def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False

async def _convert_streamed_response_to_string(response: AsyncGenerator):
    return "".join([str(chunk) async for chunk in response])

def _force_async_generator(gen: Union[Generator, AsyncGenerator]) -> AsyncGenerator:
    """
    Takes a generator, and converts it into an async generator if it is not already.
    """
    if inspect.isasyncgen(gen):
        return gen

    async def _convert_generator_to_async():
        """
        Runs each iteration of the generator in an offloaded thread, to ensure
        the main loop is not blocked, and yield to create an async generator.
        """
        FINAL_GENERATOR_VALUE = object()
        while True:
            # Note that this is the equivalent of running:
            # next(gen, FINAL_GENERATOR_VALUE) on a separate thread,
            # ensuring that if there is anything blocking in the generator,
            # it does not block the main loop.
            chunk = await to_thread.run_sync(next, gen, FINAL_GENERATOR_VALUE)
            if chunk == FINAL_GENERATOR_VALUE:
                break
            yield chunk

    return _convert_generator_to_async()

def _handle_exception():
    # Note that logger.exception logs the stacktrace, such that the user can
    # debug this error from the logs.
    logging.exception("Internal Server Error")
    raise HTTPException(status_code=500, detail="Internal Server Error")


def _intercept_exceptions_sync(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            _handle_exception()

    return inner


def _intercept_exceptions_async(func):
    async def inner(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            _handle_exception()

    return inner