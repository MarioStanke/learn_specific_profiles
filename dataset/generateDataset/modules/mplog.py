""" Functions for logging from multiprocessing processes. """

from dataclasses import dataclass
import logging
import logging.handlers

def setup_logger(name: str = None, log_file: str = None, level: int= logging.INFO, 
                 handler: logging.Handler = None) -> logging.Logger:
    """ Function to setup as many loggers as you want. With the default setup, the root logger will be set up. 
        If you want to set up a different logger, you have to specify the name. 
        If you specify a log_file, the log will _also_ be written to that file. Otherwise, the log will be written to 
        the console by default or to the specified handler via the handler argument. """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file is not None:
        fhandler = logging.FileHandler(log_file, mode="w", encoding='utf-8')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

    return logger



@dataclass
class customLogrecord:
    """ Class to store log messages in a custom queue. """
    levelno: int
    msg: str
    args: tuple = None
    kwargs: dict = None



class customBufferedLogger():
    """ Class to buffer log messages and emit them all at once to a target logger on request. 
        In principle this should be possible with logging.handlers.MemoryHandler, but it's so convoluted that it's
        easier to just write a custom class. """

    def __init__(self):
        self.queue = []

    def emit(self, logger: logging.Logger):
        for record in self.queue:
            logger.log(record.levelno, record.msg, *record.args, **record.kwargs)
        
    def debug(self, msg: str, *args, **kwargs):
        """ Log a debug message to the buffer. """
        self.queue.append(customLogrecord(logging.DEBUG, msg, args, kwargs))

    def info(self, msg: str, *args, **kwargs):
        """ Log an info message to the buffer. """
        self.queue.append(customLogrecord(logging.INFO, msg, args, kwargs))

    def warning(self, msg: str, *args, **kwargs):
        """ Log a warning message to the buffer. """
        self.queue.append(customLogrecord(logging.WARNING, msg, args, kwargs))

    def error(self, msg: str, *args, **kwargs):
        """ Log an error message to the buffer. """
        self.queue.append(customLogrecord(logging.ERROR, msg, args, kwargs))

    def critical(self, msg: str, *args, **kwargs):
        """ Log a critical message to the buffer. This immediately flushes the buffer. """
        self.queue.append(customLogrecord(logging.CRITICAL, msg, args, kwargs))



def mp_log_listener(queue):
    """ Function to listen to a queue and log the messages. The queue should be filled with 
        customBufferedLogger objects """
    root = logging.getLogger() #setup_logger(level=logging.DEBUG)
    while True:
        try:
            bufferedLog = queue.get()
            if bufferedLog is None:  # We send this as a sentinel to tell the listener to quit.
                break
            
            bufferedLog.emit(root)

        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)



import multiprocessing
def mp_process_manager(processes: list[multiprocessing.Process], p: int):
    """ Gets a list of initialized but not yet running multiprocessing.Process objects and keeps at most p processes 
        running until all are finished. """
    assert p > 0, "[ERROR] >>> p <= 0"
    running = []
    total = len(processes)
    progressstep = total // 100 if total >= 100 else 1
    finished = 0
    logging.info(f"Starting {total} processes, with {p} running simultaneously...")
    while len(processes) > 0 or len(running) > 0:
        while len(running) < p and len(processes) > 0:
            process = processes.pop()
            process.start()
            running.append(process)

        for i, process in enumerate(running):
            if process.exitcode is not None:
                process.join()
                running.pop(i)
                finished += 1
                # every 1% print progress
                if finished % progressstep == 0:
                    logging.info(f"Finished {finished}/{total} processes")

                break

    # wait for all processes to finish (should be done already)
    for p in processes:
        p.join()