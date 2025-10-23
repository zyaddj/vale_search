"""
Structured logging utilities for ValeSearch.
"""

import structlog
import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", json_logs: bool = False):
    """Configure structured logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Configure logging on import
configure_logging()