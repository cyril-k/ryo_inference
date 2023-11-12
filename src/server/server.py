import sys
import socket
import logging
import multiprocessing
import time
from typing import (
    Dict,
    List,
    Optional
)
from pathlib import Path
import uvicorn
import asyncio
import signal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from server.endpoints import GenericEndpoints
from model_handler import ModelHandler
from utils.logging import setup_logging
import utils.errors as errors
from server.utils.middleware import TerminationHandlerMiddleware
from server.utils.common import all_processes_dead

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import Response, StreamingResponse

NUM_WORKERS = 1 # TODO check if spawning multiple model instances
WORKER_TERMINATION_TIMEOUT_SECS = 120.0
WORKER_TERMINATION_CHECK_INTERVAL_SECS = 0.5

FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s [%(funcName)s():%(lineno)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

class UvicornCustomServer(multiprocessing.Process):
    def __init__(
        self, 
        config: uvicorn.Config, 
        sockets: Optional[List[socket.socket]] = None
    ) -> None:
        super().__init__()
        self.sockets = sockets
        self.config = config

    def stop(self):
        self.terminate()

    def run(self):
        server = uvicorn.Server(config=self.config)
        asyncio.run(server.serve(sockets=self.sockets))

class InferenceServer:
    def __init__(
        self,
        http_port: int,
        config: Dict,
        setup_json_logger: bool = True,
    ) -> None:
        self.http_port = http_port
        self._config = config
        self._model = ModelHandler(self._config)
        self._endpoints = GenericEndpoints(self._model)
        self._setup_json_logger = setup_json_logger

    def cleanup(self):
        if INFERENCE_SERVER_FAILED_FILE.exists():
            INFERENCE_SERVER_FAILED_FILE.unlink()

    def on_startup(self):
        """
        This method will be started inside the main process, 
        so here is where we want to setup our logging and model
        """
        self.cleanup()

        if self._setup_json_logger:
            setup_logging()

        # begin model loading
        self._model.start_load()

    def create_application(self):
        """This method creates a FastAPI application with configuration for running model inference.

        Returns:
            FastAPI: configured FastAPI application
        """
        app = FastAPI(
            title="RYO Inference Server",
            docs_url=None,
            redoc_url=None,
            default_response_class=ORJSONResponse,
            on_startup=[self.on_startup],
            routes=[
                # liveness endpoint
                APIRoute(r"/", lambda: True),
                # readiness endpoint
                APIRoute(
                    r"/status", 
                    self._endpoints.model_running, 
                    tags=["Model Access"]
                ),
                APIRoute(
                    r"/generate",
                    self._endpoints.generte,
                    methods=["POST"],
                    tags=["Model Access"],
                ),
            ],
            exception_handlers={
                errors.InferenceError: errors.inference_error_handler,
                errors.ModelNotFound: errors.model_not_found_handler,
                errors.ModelNotReady: errors.model_not_ready_handler,
                NotImplementedError: errors.not_implemented_error_handler,
                HTTPException: errors.http_exception_handler,
                Exception: errors.generic_exception_handler,
            },
        )

        def exit_self():
            # Note that this kills the current process, the worker process, not
            # the main server process.
            sys.exit()

        termination_handler_middleware = TerminationHandlerMiddleware(
            on_stop=lambda: None,
            on_term=exit_self,
        )
        app.add_middleware(BaseHTTPMiddleware, dispatch=termination_handler_middleware)
        return app

    def start(self):
        """This method configures and starts the Uvicorn server to serve the FastAPI application.
        """
        cfg = uvicorn.Config(
            self.create_application(),
            host="0.0.0.0",
            port=self.http_port,
            workers=NUM_WORKERS,
            log_config={
                "version": 1,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "datefmt": DATE_FORMAT,
                        "fmt": "%(asctime)s.%(msecs)03d %(name)s %(levelprefix)s %(message)s",
                        "use_colors": None,
                    },
                    "access": {
                        "()": "uvicorn.logging.AccessFormatter",
                        "datefmt": DATE_FORMAT,
                        "fmt": "%(asctime)s.%(msecs)03d %(name)s %(levelprefix)s %(client_addr)s %(process)s - "
                        '"%(request_line)s" %(status_code)s',
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    },
                    "access": {
                        "formatter": "access",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                },
                "loggers": {
                    "uvicorn": {"handlers": ["default"], "level": "INFO"},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {
                        "handlers": ["access"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            },
        )

        # Call this so uvloop gets used
        cfg.setup_event_loop()

        async def serve():
            """Sets up a TCP socket, starts multiple Uvicorn servers (equal to the number of workers), 
                and handles stop signals to shut down the servers gracefully.
            """
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serversocket.bind((cfg.host, cfg.port))
            serversocket.listen(5)

            logging.info(f"starting uvicorn with {cfg.workers} workers")
            servers: List[UvicornCustomServer] = []
            for _ in range(cfg.workers):
                server = UvicornCustomServer(config=cfg, sockets=[serversocket])
                server.start()
                servers.append(server)

            def stop_servers():
                # Send stop signal, then wait for all to exit
                for server in servers:
                    # Sends term signal to the process, which should be handled
                    # by the termination handler.
                    server.stop()

                termination_check_attempts = int(
                    WORKER_TERMINATION_TIMEOUT_SECS
                    / WORKER_TERMINATION_CHECK_INTERVAL_SECS
                )
                for _ in range(termination_check_attempts):
                    time.sleep(WORKER_TERMINATION_CHECK_INTERVAL_SECS)
                    if all_processes_dead(servers):
                        # Exit main process
                        sys.exit()

            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
                signal.signal(sig, lambda sig, frame: stop_servers())

        async def servers_task():
            servers = [serve()]
            await asyncio.gather(*servers)

        asyncio.run(servers_task())
