# Centralized logging for the Financial AI Assistant.
# DEBUG for dev, INFO for production, WARNING for potential issues, ERROR for actual problems.
# using python's built-in logging module for flexibility and performance.

import logging
import sys
from app.config.settings import settings

# __name__ is the name of the module, so logs will show which module they came from. (matches file name)
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Guard: if this logger already has handlers, it was already configured.
    # Without this check, calling get_logger() twice would double every log line.
    if logger.handlers:
        return logger  

    # level: Convert the string from settings ("INFO", "DEBUG", etc.) to the
    # integer constant that Python's logging module expects (20, 10, etc.)
    level = logging.getLevelName(settings.LOG_LEVEL.upper())
    logger.setLevel(level)

    # handler: Output logs to standard output (console).
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # formatter: Define the format of log messages.
    # Fields:
    #   %(asctime)s   → timestamp (when did this happen?)
    #   %(levelname)  → INFO / DEBUG / WARNING / ERROR (how serious?)
    #   %(name)       → module path (where in the code?)
    #   %(message)s   → the actual log message (what happened?)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
 
    # Attach the handler to this logger
    logger.addHandler(handler)
 
    # Prevent propagation to root logger 
    # Python loggers form a hierarchy: "app.ingestion.loader" → "app.ingestion"
    # → "app" → root. By default a log record bubbles up to every ancestor.
    # If the root logger also has a handler (common in notebooks/frameworks),
    # you'd see every line printed twice. This stops the bubbling.
    logger.propagate = False

    return logger