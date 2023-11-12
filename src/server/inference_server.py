import os
from typing import Dict

import yaml
from components.server import InferenceServer
from utils.logging import setup_logging

CONFIG_FILE = "config.yaml"

setup_logging()


class RunningServer:
    _config: Dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        server = InferenceServer(http_port=self._port, config=self._config)
        server.start()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    RunningServer(CONFIG_FILE, env_port).start()