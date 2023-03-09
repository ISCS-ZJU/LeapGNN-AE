import argparse
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

def setup_primary_logging(log_file_path: str, error_log_file_path: str) -> Queue:
    """
    Global logging is setup using this method. In a distributed setup, a multiprocessing queue is setup
    which can be used by the workers to write their log messages. This initializers respective handlers
    to pick messages from the queue and handle them to write to corresponding output buffers.
    Parameters
    ----------
    log_file_path : ``str``, required
        File path to write output log
    error_log_file_path: ``str``, required
        File path to write error log
    Returns
    -------
    log_queue : ``torch.multiprocessing.Queue``
        A log queue to which the log handler listens to. This is used by workers
        in a distributed setup to initialize worker specific log handlers(refer ``setup_worker_logging`` method).
        Messages posted in this queue by the workers are picked up and bubbled up to respective log handlers.
    """
    # Multiprocessing queue to which the workers should log their messages
    log_queue = Queue(-1)

    # Handlers for stream/file logging
    output_file_log_handler = logging.FileHandler(filename=str(log_file_path))
    error_file_log_handler = logging.FileHandler(filename=str(error_log_file_path))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    output_file_log_handler.setFormatter(formatter)
    error_file_log_handler.setFormatter(formatter)

    output_file_log_handler.setLevel(logging.INFO)
    error_file_log_handler.setLevel(logging.ERROR)

    # This listener listens to the `log_queue` and pushes the messages to the list of
    # handlers specified.
    listener = QueueListener(log_queue, output_file_log_handler, error_file_log_handler, respect_handler_level=True)

    listener.start()

    return log_queue

def setup_worker_logging(rank: int, log_queue: Queue):
    """
    Method to initialize worker's logging in a distributed setup. The worker processes
    always write their logs to the `log_queue`. Messages in this queue in turn gets picked
    by parent's `QueueListener` and pushes them to respective file/stream log handlers.
    Parameters
    ----------
    rank : ``int``, required
        Rank of the worker
    log_queue: ``Queue``, required
        The common log queue to which the workers
    Returns
    -------
    features : ``np.ndarray``
        The corresponding log power spectrogram.
    """
    queue_handler = QueueHandler(log_queue)

    # Add a filter that modifies the message to put the
    # rank in the log format
    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    # Default logger level is WARNING, hence the change. Otherwise, any worker logs
    # are not going to get bubbled up to the parent's logger handlers from where the
    # actual logs are written to the output
    root_logger.setLevel(logging.INFO)

class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True