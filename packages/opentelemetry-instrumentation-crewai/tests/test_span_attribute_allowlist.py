"""
Span attributes come from a fixed set of fields, never a __dict__ walk, so an
object we don't control (an LLM client, an embedder config) can't leak its
credentials through its repr.

Objects are constructed only -- no kickoff, no network.
"""

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai.tools import BaseTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.crewai.crewai_span_attributes import CrewAISpanAttributes

# Not a credential: an opaque marker with no provider shape, used only to prove
# that whatever is configured on an LLM or embedder stays off the span.
SENTINEL = "SENTINEL-NOT-A-KEY-9f3a"


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echoes the input back."

    def _run(self, text: str = "") -> str:
        return text


def build_agent():
    return Agent(
        role="researcher",
        goal="find things",
        backstory="a fixed backstory",
        llm=LLM(model="test-model", api_key=SENTINEL),
        embedder={"provider": "openai", "config": {"api_key": SENTINEL}},
        tools=[EchoTool()],
    )


def build_task():
    return Task(description="a fixed description", expected_output="a fixed output",
                agent=build_agent(), tools=[EchoTool()])


def build_crew():
    agent = build_agent()
    return Crew(
        agents=[agent],
        tasks=[Task(description="a fixed description", expected_output="a fixed output", agent=agent)],
        name="fixed-crew",
        manager_llm=LLM(model="test-model", api_key=SENTINEL),
        embedder={"provider": "openai", "config": {"api_key": SENTINEL}},
    )


BUILDERS = {"Agent": build_agent, "Task": build_task, "Crew": build_crew}


def span_attributes(instance):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(__name__).start_as_current_span("test") as span:
        CrewAISpanAttributes(span=span, instance=instance)
    return dict(exporter.get_finished_spans()[0].attributes)


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_configured_credentials_never_reach_the_span(kind):
    attrs = span_attributes(BUILDERS[kind]())

    # Substring check: nested agents, tasks and tools are JSON-dumped into a
    # single value, so a key-wise check would miss anything hidden inside them.
    for key, value in attrs.items():
        assert SENTINEL not in str(value), f"{key} carries configured LLM state"

    # ...while the allowlisted fields are still emitted.
    assert attrs[f"crewai.{kind.lower()}.id"]
