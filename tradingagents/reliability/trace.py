"""No-op tracer stub. Future hook for LangSmith/OpenTelemetry."""
from contextlib import contextmanager
from typing import Iterator


class _NoopTracer:
    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def event(self, name: str, **attrs) -> None:
        pass


_tracer = _NoopTracer()


def get_tracer() -> _NoopTracer:
    return _tracer
