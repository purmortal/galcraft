import logging


# class Logger:
#     def __init__(self, logfile, logger=None, level=logging.INFO, mode='a'):
#         self.logger = logging.getLogger(logger)
#         self.logger.propagate = False
#         self.logger.setLevel(level)
#         fh = logging.FileHandler(logfile, mode=mode, encoding='utf-8')
#         fh.setLevel(level)
#         sh = logging.StreamHandler()
#         sh.setLevel(level)
#         formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
#         fh.setFormatter(formatter)
#         sh.setFormatter(formatter)
#         self.logger.handlers.clear()
#         self.logger.addHandler(fh)
#         self.logger.addHandler(sh)
#         fh.close()
#         sh.close()
#
#     def get_log(self):
#         return self.logger


import logging
import sys


class Logger:
    def __init__(self, logfile, name=None, level=logging.INFO, mode='a', to_stdout=True):
        """
        Parameters
        ----------
        logfile : str
            Path to the log file.
        name : str | None
            Logger name. None means a module-level logger, not the root.
        level : int
            Logging level, e.g., logging.INFO.
        mode : str
            File mode for the log file ('a' append, 'w' overwrite).
        to_stdout : bool
            If True, stream to stdout; otherwise stderr.
        """
        # Use a named logger (avoid root unless you really want it)
        self.logger = logging.getLogger(name or __name__)
        self.logger.setLevel(level)
        self.logger.propagate = False  # keep it self-contained

        # If you're recreating the logger, clear previous handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

        # File handler
        self.fh = logging.FileHandler(logfile, mode=mode, encoding='utf-8', delay=True)
        self.fh.setLevel(level)
        self.fh.setFormatter(fmt)
        self.logger.addHandler(self.fh)

        # Stream handler (to terminal)
        stream = sys.stdout if to_stdout else sys.stderr
        self.sh = logging.StreamHandler(stream=stream)
        self.sh.setLevel(level)
        self.sh.setFormatter(fmt)
        self.logger.addHandler(self.sh)

    def get_log(self):
        return self.logger

    def close(self):
        """Cleanly detach and close handlers (call at shutdown if desired)."""
        for h in list(self.logger.handlers):
            h.flush()
            h.close()
            self.logger.removeHandler(h)
