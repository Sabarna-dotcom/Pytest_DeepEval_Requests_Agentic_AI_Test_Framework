import logging
from pathlib import Path

class LogGen():

    @staticmethod
    def loggen():
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / "automation.log").resolve()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_file
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s: %(levelname)s: %(message)s",
                datefmt="%m/%d/%Y %I:%M:%S %p"
            ))
            logger.addHandler(file_handler)

        return logger