"""Shared utilities for MCP instrumentation."""

import asyncio
import logging
import os
import traceback


class Config:
    """Module-level configuration for the MCP instrumentation."""
    exception_logger = None


def should_send_prompts() -> bool:
    """Whether request/response content may be recorded on spans.

    Mirrors the traceloop SDK's ``TRACELOOP_TRACE_CONTENT`` switch: content
    capture is on unless an operator explicitly turns it off. Shared by the
    FastMCP server wrapper and the MCP client path so a single environment
    variable governs both, which is what the package README documents.
    """
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        """Await the wrapped coroutine, logging instead of raising on failure."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        """Call the wrapped function, logging instead of raising on failure."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        """Log a tracing failure and hand it to the configured exception logger."""
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
