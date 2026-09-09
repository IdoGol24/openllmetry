"""TRACELOOP_TRACE_CONTENT must gate the MCP client path, not only FastMCP.

The package documents TRACELOOP_TRACE_CONTENT as the switch that disables content
logging. Before this test, only the FastMCP server-side wrapper consulted it: the
client path (tools/call arguments, non-tool request bodies, response bodies) recorded
content regardless, so an operator who turned the switch off still got request and
response payloads on their spans.

Each test drives the real client wrapper with a marker value and asserts the marker
is absent from every span attribute when content capture is off, and present when it
is on, so the test fails if either the gate or the capture itself regresses.
"""

import json

from fastmcp import Client, FastMCP

MARKER = "content-capture-marker-9f3a"


def _all_attribute_text(span_exporter) -> str:
    """Every attribute value across every exported span, as one string."""
    chunks = []
    for span in span_exporter.get_finished_spans():
        for value in (span.attributes or {}).values():
            if isinstance(value, (list, tuple)):
                chunks.extend(str(item) for item in value)
            else:
                chunks.append(str(value))
    return "\n".join(chunks)


def _server() -> FastMCP:
    server = FastMCP("content-gate-server")

    @server.tool()
    async def echo_secret(token: str) -> str:
        """Echo back a caller-supplied token."""
        return f"received {token}"

    return server


async def test_tool_arguments_suppressed_when_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    assert span_exporter.get_finished_spans(), "expected the tool call to be traced"
    assert MARKER not in _all_attribute_text(span_exporter)


async def test_tool_arguments_captured_when_content_capture_on(
    span_exporter, monkeypatch
) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    # The gate must not silently disable capture altogether: with the switch on,
    # the argument is still recorded.
    assert MARKER in _all_attribute_text(span_exporter)


async def test_non_tool_response_body_suppressed_when_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """list_tools goes through _handle_mcp_method, which serialized the whole response.

    The marker lives in the registered tool description, so it travels back in the
    list_tools result and exercises the response-serialization path.
    """
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    server = FastMCP("content-gate-server")

    @server.tool(description=f"A tool whose description carries {MARKER}.")
    async def documented(arg: str) -> str:
        return arg

    async with Client(server) as client:
        tools = await client.list_tools()

    assert any(MARKER in (t.description or "") for t in tools), (
        "the marker must reach the client, otherwise this test proves nothing"
    )
    assert span_exporter.get_finished_spans(), "expected the request to be traced"
    assert MARKER not in _all_attribute_text(span_exporter)


async def test_span_structure_survives_content_capture_off(
    span_exporter, monkeypatch
) -> None:
    """Turning content off must not remove spans or their non-content attributes."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    async with Client(_server()) as client:
        await client.call_tool("echo_secret", {"token": MARKER})

    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.endswith(".tool")]
    assert tool_spans, f"expected a tool span, got {[s.name for s in spans]}"

    entity_names = [
        (s.attributes or {}).get("traceloop.entity.name") for s in tool_spans
    ]
    assert "echo_secret" in entity_names

    # Structural attributes stay; only content is withheld.
    for span in tool_spans:
        attributes = span.attributes or {}
        assert "traceloop.span.kind" in attributes
        for key in ("traceloop.entity.input", "traceloop.entity.output"):
            if key in attributes:
                json.loads(attributes[key])  # if present it must still be valid JSON
                assert MARKER not in attributes[key]
