"""
Logging utility for REQreate instance generator.
Creates clean, informative logs both to console and file.
"""

import logging
import os
from datetime import datetime


class InstanceLogger:
    """Logger that writes to both console and file in the network directory."""
    
    def __init__(self, place_name, log_dir=None):
        self.place_name = place_name
        self.logger = logging.getLogger('REQreate')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter('%(message)s')
        
        # Console handler (clean output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (detailed output)
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), place_name, 'logs')
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'generation_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
        
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(f"⚠️  {message}")
    
    def error(self, message):
        """Log error message."""
        self.logger.error(f"❌ {message}")
    
    def success(self, message):
        """Log success message."""
        self.logger.info(f"✓ {message}")
    
    def progress(self, message):
        """Log progress message."""
        self.logger.info(f"  → {message}")
    
    def section(self, title):
        """Log section header."""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"{title}")
        self.logger.info(separator)
    
    def subsection(self, title):
        """Log subsection header."""
        self.logger.info(f"\n{title}")
        self.logger.info("-" * 40)


# Global logger instance
_global_logger = None


def get_logger(place_name=None):
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None and place_name is not None:
        _global_logger = InstanceLogger(place_name)
    return _global_logger


def init_logger(place_name):
    """Initialize the logger for a new generation session."""
    global _global_logger
    _global_logger = InstanceLogger(place_name)
    return _global_logger
