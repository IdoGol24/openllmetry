from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple, Union, cast
import json
import logging

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.mcp.version import __version__
from opentelemetry.instrumentation.mcp.utils import (
    Config,
    dont_throw,
    should_send_prompts,
)
from opentelemetry.instrumentation.mcp.fastmcp_instrumentation import (
    FastMCPInstrumentor,
)

_instruments = ("mcp >= 1.6.0",)


class McpInstrumentor(BaseInstrumentor):
    """Instrument the MCP client, server and transports with OpenTelemetry spans."""
    def __init__(self, exception_logger=None):
        """Store the exception logger and build the FastMCP sub-instrumentor."""
        super().__init__()
        Config.exception_logger = exception_logger
        self._fastmcp_instrumentor = FastMCPInstrumentor()

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the package versions this instrumentation supports."""
        return _instruments

    def _instrument(self, **kwargs):
        """Wrap the MCP client, server sessions and every supported transport."""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Instrument FastMCP
        self._fastmcp_instrumentor.instrument(tracer)

        # Instrument FastMCP Client to create a session-level span
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aenter__",
                self._fastmcp_client_enter_wrapper(tracer),
            ),
            "fastmcp.client",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aexit__",
                self._fastmcp_client_exit_wrapper(tracer),
            ),
            "fastmcp.client",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse",
                "SseServerTransport.connect_sse",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._transport_wrapper(tracer)
            ),
            "mcp.server.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session",
                "ServerSession.__init__",
                self._base_session_init_wrapper(tracer),
            ),
            "mcp.server.session",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.streamable_http",
                "streamablehttp_client",
                self._transport_wrapper(tracer),
            ),
            "mcp.client.streamable_http",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.streamable_http",
                "StreamableHTTPServerTransport.connect",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.streamable_http",
        )
        wrap_function_wrapper(
            "mcp.shared.session",
            "BaseSession.send_request",
            self.patch_mcp_client(tracer),
        )

    def _uninstrument(self, **kwargs):
        """Unwrap the transports this instrumentation replaced."""
        unwrap("mcp.client.stdio", "stdio_client")
        unwrap("mcp.server.stdio", "stdio_server")
        self._fastmcp_instrumentor.uninstrument()

    def _transport_wrapper(self, tracer):
        """Wrap a transport so its read and write streams are instrumented."""
        @asynccontextmanager
        async def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
        ) -> AsyncGenerator[
            Union[
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter],
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter, Any],
            ],
            None,
        ]:
            """Yield the transport's streams wrapped in instrumented proxies."""
            async with wrapped(*args, **kwargs) as result:
                try:
                    read_stream, write_stream = result
                    yield InstrumentedStreamReader(
                        read_stream, tracer
                    ), InstrumentedStreamWriter(write_stream, tracer)
                except ValueError:
                    try:
                        read_stream, write_stream, get_session_id_callback = result
                        yield InstrumentedStreamReader(
                            read_stream, tracer
                        ), InstrumentedStreamWriter(
                            write_stream, tracer
                        ), get_session_id_callback
                    except Exception as e:
                        logging.warning(
                            f"mcp instrumentation _transport_wrapper exception: {e}"
                        )
                        yield result
                except Exception as e:
                    logging.warning(
                        f"mcp instrumentation transport_wrapper exception: {e}"
                    )
                    yield result

        return traced_method

    def _base_session_init_wrapper(self, tracer):
        """Wrap a server session's incoming message streams to carry trace context."""
        def traced_method(
            wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
        ) -> None:
            """Replace the session's incoming stream pair with context-propagating proxies."""
            wrapped(*args, **kwargs)
            reader = getattr(instance, "_incoming_message_stream_reader", None)
            writer = getattr(instance, "_incoming_message_stream_writer", None)
            if reader and writer:
                setattr(
                    instance,
                    "_incoming_message_stream_reader",
                    ContextAttachingStreamReader(reader, tracer),
                )
                setattr(
                    instance,
                    "_incoming_message_stream_writer",
                    ContextSavingStreamWriter(writer, tracer),
                )

        return traced_method

    def patch_mcp_client(self, tracer: Tracer):
        """Wrap BaseSession.send_request so each MCP request becomes a span."""
        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            """Start a span for the outgoing request and inject trace context into its meta."""
            meta = None
            method = None
            params = None
            if len(args) > 0 and hasattr(args[0].root, "method"):
                method = args[0].root.method
            if len(args) > 0 and hasattr(args[0].root, "params"):
                params = args[0].root.params
            if params:
                if hasattr(args[0].root.params, "meta"):
                    meta = args[0].root.params.meta

            # Handle trace context propagation
            if meta and len(args) > 0:
                carrier = {}
                TraceContextTextMapPropagator().inject(carrier)
                meta.traceparent = carrier["traceparent"]
                args[0].root.params.meta = meta

            # Create different span types based on method
            if method == "tools/call":
                return await self._handle_tool_call(
                    tracer, method, params, args, kwargs, wrapped
                )
            else:
                return await self._handle_mcp_method(
                    tracer, method, args, kwargs, wrapped
                )

        return traced_method

    def _fastmcp_client_enter_wrapper(self, tracer):
        """Wrapper for FastMCP Client.__aenter__ to start a session trace"""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            # Start a root span for the MCP client session and make it current
            """Wrap a FastMCP client session enter to open a session span."""
            span_context_manager = tracer.start_as_current_span("mcp.client.session")
            span = span_context_manager.__enter__()
            span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "session")
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.client.session"
            )

            # Store the span context manager on the instance to properly exit it later
            setattr(instance, "_tracing_session_context_manager", span_context_manager)

            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.set_attribute(ERROR_TYPE, type(e).__name__)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

        return traced_method

    def _fastmcp_client_exit_wrapper(self, tracer):
        """Wrapper for FastMCP Client.__aexit__ to end the session trace"""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            """Close the session span when the FastMCP client exits."""
            try:
                # Call the original method first
                result = await wrapped(*args, **kwargs)

                # End the session span context manager
                context_manager = getattr(
                    instance, "_tracing_session_context_manager", None
                )
                if context_manager:
                    context_manager.__exit__(None, None, None)

                return result
            except Exception as e:
                # End the session span context manager with exception info
                context_manager = getattr(
                    instance, "_tracing_session_context_manager", None
                )
                if context_manager:
                    context_manager.__exit__(type(e), e, e.__traceback__)
                raise

        return traced_method

    async def _handle_tool_call(self, tracer, method, params, args, kwargs, wrapped):
        """Handle tools/call with tool semantics"""
        # Extract the actual tool name
        entity_name = method
        span_name = f"{method}.tool"
        if params:
            try:
                if hasattr(params, "name"):
                    entity_name = params.name
                    span_name = f"{params.name}.tool"
                elif hasattr(params, "__dict__") and "name" in params.__dict__:
                    entity_name = params.__dict__["name"]
                    span_name = f"{params.__dict__['name']}.tool"
            except Exception:
                pass

        with tracer.start_as_current_span(span_name) as span:
            # Set tool-specific attributes
            span.set_attribute(
                SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value
            )
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

            # Add input. Tool arguments are request content, so they are
            # recorded only when content capture is enabled.
            clean_input = (
                self._extract_clean_input(method, params)
                if should_send_prompts()
                else None
            )
            if clean_input:
                try:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(clean_input)
                    )
                except (TypeError, ValueError):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, str(clean_input)
                    )

            return await self._execute_and_handle_result(
                span, method, args, kwargs, wrapped, clean_output=True
            )

    async def _handle_mcp_method(self, tracer, method, args, kwargs, wrapped):
        """Handle non-tool MCP methods with simple serialization"""
        with tracer.start_as_current_span(f"{method}.mcp") as span:
            # The serialized request is content: it carries caller-supplied
            # params, so it is recorded only when content capture is enabled.
            if should_send_prompts():
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, f"{serialize(args[0])}"
                )
            return await self._execute_and_handle_result(
                span, method, args, kwargs, wrapped, clean_output=False
            )

    async def _execute_and_handle_result(
        self, span, method, args, kwargs, wrapped, clean_output=False
    ):
        """Execute the wrapped function and handle the result"""
        try:
            result = await wrapped(*args, **kwargs)
            # Add output. The response body is content, so it is recorded only
            # when content capture is enabled.
            if not should_send_prompts():
                pass
            elif clean_output:
                clean_output_data = self._extract_clean_output(method, result)
                if clean_output_data:
                    try:
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            json.dumps(clean_output_data),
                        )
                    except (TypeError, ValueError):
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            str(clean_output_data),
                        )
            else:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, serialize(result)
                )
            # Handle errors
            if hasattr(result, "isError") and result.isError:
                span.set_attribute(ERROR_TYPE, "tool_error")
                if len(result.content) > 0:
                    span.set_status(
                        Status(StatusCode.ERROR, f"{result.content[0].text}")
                    )
            else:
                span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def _extract_clean_input(self, method: str, params: Any) -> dict:
        """Extract clean input parameters for different MCP method types"""
        if not params:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract name and arguments
                result = {}
                if hasattr(params, "name"):
                    result["tool_name"] = params.name
                if hasattr(params, "arguments"):
                    if hasattr(params.arguments, "__dict__"):
                        result["arguments"] = params.arguments.__dict__
                    else:
                        result["arguments"] = params.arguments
                elif hasattr(params, "__dict__") and "arguments" in params.__dict__:
                    result["arguments"] = params.__dict__["arguments"]
                return result
            elif method == "tools/list":
                # For list_tools, there are usually no parameters
                return {}
            else:
                # For other methods, try to serialize params cleanly
                if hasattr(params, "__dict__"):
                    # Remove internal fields starting with _ and non-serializable objects
                    clean_params = {}
                    for k, v in params.__dict__.items():
                        if not k.startswith("_"):
                            try:
                                # Test if the value is JSON serializable
                                json.dumps(v)
                                clean_params[k] = v
                            except (TypeError, ValueError):
                                # If not serializable, store a string representation
                                clean_params[k] = str(type(v).__name__)
                    return clean_params
                else:
                    return {"params": str(params)}
        except Exception:
            return {}

    def _extract_clean_output(self, method: str, result: Any) -> dict:
        """Extract clean output for different MCP method types"""
        if not result:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract the actual result content
                output = {}
                if hasattr(result, "content") and result.content:
                    if len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            output["result"] = content_item.text
                        elif hasattr(content_item, "__dict__"):
                            output["result"] = content_item.__dict__
                        else:
                            output["result"] = str(content_item)

                # Check if this is an error response
                if hasattr(result, "isError") and result.isError:
                    output["is_error"] = True

                return output
            elif method == "tools/list":
                # For list_tools, extract tool names and descriptions
                output = {"tools": []}
                if hasattr(result, "tools") and result.tools:
                    for tool in result.tools:
                        tool_info = {}
                        if hasattr(tool, "name"):
                            tool_info["name"] = tool.name
                        if hasattr(tool, "description"):
                            tool_info["description"] = tool.description
                        output["tools"].append(tool_info)
                return output
            else:
                # For other methods, try to serialize result cleanly
                if hasattr(result, "__dict__"):
                    clean_result = {
                        k: v
                        for k, v in result.__dict__.items()
                        if not k.startswith("_")
                    }
                    return clean_result
                else:
                    return {"result": str(result)}
        except Exception:
            return {}


def serialize(request, depth=0, max_depth=4):
    """Serialize input args to MCP server into JSON.
    The function accepts input object and converts into JSON
    keeping depth in mind to prevent creating large nested JSON"""
    if depth > max_depth:
        return {}
    depth += 1

    def is_serializable(request):
        """Return whether a value can be JSON-encoded without a fallback."""
        try:
            json.dumps(request)
            return True
        except Exception:
            return False

    if is_serializable(request):
        return json.dumps(request)
    else:
        result = {}
        try:
            if hasattr(request, "model_dump_json"):
                return request.model_dump_json()
            if hasattr(request, "__dict__"):
                for attrib in request.__dict__:
                    if not attrib.startswith("_"):
                        if type(request.__dict__[attrib]) in [
                            bool,
                            str,
                            int,
                            float,
                            type(None),
                        ]:
                            result[str(attrib)] = request.__dict__[attrib]
                        else:
                            result[str(attrib)] = serialize(
                                request.__dict__[attrib], depth
                            )
        except Exception:
            pass
        return json.dumps(result)


class InstrumentedStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    """Stream reader proxy that extracts trace context from incoming messages."""
    def __init__(self, wrapped, tracer):
        """Wrap a stream reader and keep the tracer used for its spans."""
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        """Enter the wrapped stream reader."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the wrapped stream reader."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Iterate the wrapped reader, restoring trace context from each message."""
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        async for item in self.__wrapped__:
            # Handle different item types based on what's available
            request = None
            if hasattr(item, "message") and hasattr(item.message, "root"):
                request = item.message.root
            elif type(item) is JSONRPCMessage:
                request = cast(JSONRPCMessage, item).root
            elif hasattr(item, "root"):
                request = item.root
            else:
                yield item
                continue

            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    """Stream writer proxy that records outgoing responses on a span."""
    def __init__(self, wrapped, tracer):
        """Wrap a stream writer and keep the tracer used for its spans."""
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        """Enter the wrapped stream writer."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the wrapped stream writer."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        """Record the outgoing response on a span and forward it to the wrapped stream."""
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        # Handle different item types based on what's available
        request = None
        if hasattr(item, "message") and hasattr(item.message, "root"):
            request = item.message.root
        elif type(item) is JSONRPCMessage:
            request = cast(JSONRPCMessage, item).root
        elif hasattr(item, "root"):
            request = item.root
        else:
            return await self.__wrapped__.send(item)

        with self._tracer.start_as_current_span("ResponseStreamWriter") as span:
            if hasattr(request, "result"):
                # The response body is content; the error status below is not,
                # so only the value itself is gated.
                if should_send_prompts():
                    span.set_attribute(
                        SpanAttributes.MCP_RESPONSE_VALUE,
                        f"{serialize(request.result)}",
                    )
                if "isError" in request.result:
                    if request.result["isError"] is True:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                f"{request.result['content'][0]['text']}",
                            )
                        )
            if hasattr(request, "id"):
                span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{request.id}")

            if not isinstance(request, JSONRPCRequest):
                return await self.__wrapped__.send(item)
            meta = None
            if not request.params:
                request.params = {}
            meta = request.params.setdefault("_meta", {})

            propagate.get_global_textmap().inject(meta)
            return await self.__wrapped__.send(item)


@dataclass(slots=True, frozen=True)
class ItemWithContext:
    """A stream item paired with the OpenTelemetry context it was written under."""
    item: Any
    ctx: context.Context


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    """Stream writer proxy that attaches the current context to each item."""
    def __init__(self, wrapped, tracer):
        """Wrap a stream writer and keep the tracer used for its spans."""
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        """Enter the wrapped stream writer."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the wrapped stream writer."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        # Removed RequestStreamWriter span creation - we don't need low-level protocol spans
        """Forward the item together with the context it was sent under."""
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    """Stream reader proxy that restores each item's saved context while it is handled."""
    def __init__(self, wrapped, tracer):
        """Wrap a stream reader and keep the tracer used for its spans."""
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        """Enter the wrapped stream reader."""
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        """Exit the wrapped stream reader."""
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        """Yield each item with its saved context attached, detaching it afterwards."""
        async for item in self.__wrapped__:
            item_with_context = cast(ItemWithContext, item)
            restore = context.attach(item_with_context.ctx)
            try:
                yield item_with_context.item
            finally:
                context.detach(restore)
