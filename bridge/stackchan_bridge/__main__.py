import asyncio
import logging
import time

from . import config
from .server import serve_forever


def main() -> None:
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(serve_forever(config.load()))


if __name__ == "__main__":
    main()
