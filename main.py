import sys

from application import Application
import logging

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)], force=True)


if __name__ == "__main__":
    app = Application()
    app.run()
