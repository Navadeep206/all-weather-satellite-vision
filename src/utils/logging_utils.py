import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[str] = "INFO") -> logging.Logger:
    """Creates or retrieves a logger with console output and standardized formatting.
    
    Args:
        name (str): Name of the module requesting the logger (typically __name__).
        level (str, optional): Logging level name (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Map string level to logging level integer
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Prevent handler duplication if logger is already configured
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Clean production/research formatting
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
