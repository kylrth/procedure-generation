import random
import socket
import tempfile

import weaviate
from weaviate.embedded import EmbeddedOptions


class NiceWeaviate(weaviate.WeaviateAsyncClient):
    """This Weaviate client keeps things in a temporary directory and deletes it when done.

    It also sets nice defaults for using an embedded instance like this.
    """

    tdir: tempfile.TemporaryDirectory
    port: int
    grpc_port: int

    def __init__(self):
        self.tdir = tempfile.TemporaryDirectory()
        self.port = _find_free_port()
        self.grpc_port = _find_free_port()

    async def __aenter__(self):
        path = self.tdir.__enter__()
        super().__init__(
            embedded_options=EmbeddedOptions(
                persistence_data_path=path,
                version="1.25.17",
                port=self.port,
                additional_env_vars={
                    "AUTOSCHEMA_ENABLED": "false",
                    "DISABLE_TELEMETRY": "true",
                    "LOG_LEVEL": "warning",
                },
                grpc_port=self.grpc_port,
            )
        )

        await super().__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await super().__aexit__(exc_type, exc_value, traceback)
        self.tdir.__exit__(exc_type, exc_value, traceback)


def _find_free_port(lowest=1024, highest=65535) -> int:
    """Find a random free port to use between `lowest` and `highest`, inclusive."""
    retries = 20

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(retries):
            port = random.randint(lowest, highest)
            try:
                s.bind(("", port))
            except OSError:
                continue
            else:
                return port

    raise OSError(f"failed to find open port after {retries} tries")
