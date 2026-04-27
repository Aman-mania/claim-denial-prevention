"""
Logging Configuration
======================
Centralised structlog setup for the Claim Denial Prevention System.

Call `setup_logging()` once at each application entry point:
- run_ingestion.py   → setup_logging(level="INFO")
- tests/conftest.py  → setup_logging(level="WARNING")   (reduce noise)
- FastAPI startup    → setup_logging(level="INFO", json_output=True)  [Week 7]

Extending in later weeks
-------------------------
- Week 7 (FastAPI): set json_output=True for structured CloudWatch-compatible logs.
- Week 8 (AWS):     swap PrintLoggerFactory for a file handler or CloudWatch sink.
- All modules continue to use `structlog.get_logger(__name__)` — no changes needed.
"""

import logging
import structlog


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """
    Configure structlog processors and stdlib logging bridge.

    Parameters
    ----------
    level       : Log level ("DEBUG", "INFO", "WARNING", "ERROR").
    json_output : True → JSON renderer (production / FastAPI).
                  False → ConsoleRenderer (local development, default).
    """
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging (e.g. from pandas, pyarrow) through structlog.
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
