import os
from pathlib import Path
import signal
import json

from typing import (
    Dict,
    Union,
    Generator,
    AsyncGenerator,
)
from fastapi import Depends, HTTPException, Request
from starlette.requests import ClientDisconnect
from starlette.responses import Response, StreamingResponse

from model_handler import ModelHandler
from utils.common import transform_keys

INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()

async def parse_body(request: Request) -> bytes:
    """
    Used by FastAPI to read body in an asynchronous manner
    """
    try:
        return await request.body()
    except ClientDisconnect as exc:
        raise HTTPException(status_code=499, detail="Client disconnected") from exc
    
class DeepNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(DeepNumpyEncoder, self).default(obj)

class GenericEndpoints:

    def __init__(
            self,
            model: ModelHandler
    ) -> None:
        self._model = model

    @staticmethod
    def healthcheck(
            model: ModelHandler
        ) -> None:

        if model.load_failed():
            INFERENCE_SERVER_FAILED_FILE.touch()
            os.kill(os.getpid(), signal.SIGKILL)

        if not model.ready:
            raise RuntimeError(f"Model with name {model.name} is not ready.")

    async def model_running(self) -> Dict[str, Union[str, bool]]:
        self.healthcheck(self._model)

        return {}
    
    async def generate(
        self, 
        request: Request, 
        body_raw: bytes = Depends(parse_body)
        ) -> Response:
        """
        This method calls the user-provided predict method
        """

        self.healthcheck(self._model)

        try:
            body = json.loads(body_raw)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON payload: {str(e)}"
            )

        # calls ModelWrapper.__call__, which runs validate, preprocess, predict, and postprocess
        response: Union[Dict, Generator] = await self._model(
            body,
            headers=transform_keys(request.headers, lambda key: key.lower()),
        )

        # In the case that the model returns a Generator object, return a
        # StreamingResponse instead.
        response_headers = {}

        if isinstance(response, (AsyncGenerator, Generator)):
            response_headers.update({"X-Accel-Buffering": "no"})
            # media_type in StreamingResponse sets the Content-Type header
            return StreamingResponse(
                response, 
                media_type="application/octet-stream",
                headers=response_headers,
            )

        response_headers["Content-Type"] = "application/json"
        return Response(
            content=json.dumps(response, cls=DeepNumpyEncoder),
            headers=response_headers,
        )
