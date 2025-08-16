# logger.py
import logging
from tqdm import tqdm
from multiprocessing import Manager

_logger_instance = None
_progress_bars = None

class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # keeps tqdm bar intact
            self.flush()
        except Exception:
            self.handleError(record)

class ProgressTracker:
    def __init__(self, total_tasks=0):
        self.manager = Manager()
        self.progress = self.manager.dict()
        self.total_tasks = total_tasks
        self.lock = self.manager.Lock()
        
    def update(self, task_id, increment=1):
        with self.lock:
            self.progress[task_id] = self.progress.get(task_id, 0) + increment
            
    def get_progress(self):
        return sum(self.progress.values()) / self.total_tasks * 100 if self.total_tasks > 0 else 0

def setup_logger(log_file="app.log", name="SiKabayan"):
    global _logger_instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    _logger_instance = logger
    return logger

def get_logger(name="SiKabayan"):
    global _logger_instance
    if _logger_instance is None:
        return setup_logger(name=name)
    return _logger_instance

def init_progress(total_tasks):
    global _progress_bars
    _progress_bars = ProgressTracker(total_tasks)

def get_progress():
    global _progress_bars
    if _progress_bars is None:
        return 0
    return _progress_bars.get_progress()

def update_progress(task_id, increment=1):
    global _progress_bars
    if _progress_bars is not None:
        _progress_bars.update(task_id, increment)