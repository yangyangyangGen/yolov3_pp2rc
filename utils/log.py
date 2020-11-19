import logging
from functools import partial

__all__ = ["set_logging_config"]


def set_logging_config(log_file: str = "",
                       level=logging.DEBUG,
                       LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                       DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S %p"):

    assert level in (logging.INFO, logging.DEBUG,
                     logging.WARNING, logging.CRITICAL), ""

    config_fn = partial(logging.basicConfig, filename=log_file) \
        if "" != log_file else partial(logging.basicConfig)
    config_fn(level=logging.INFO, format=DATE_FORMAT, datefmt=DATE_FORMAT)
