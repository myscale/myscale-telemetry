from uuid import UUID
from datetime import datetime
from typing import List, Optional, Any, Dict


class SpanData:
    """Represents a span of data for tracing and monitoring.

    This class encapsulates the details of a span, which is
    a unit of work within a trace. Spans are used to track the
    timing, metadata, and status of various operations in a LLM application.

    Attributes:
        trace_id (UUID): The unique identifier for the trace this span belongs to.
        span_id (UUID): The unique identifier for this span.
        parent_span_id (Any): The identifier for the parent span, if any.
        start_time (datetime): The start time of the span.
        name (str): The name of the span, typically describing the operation.
        kind (str): The kind of span (e.g., llm, retriever).
        service_name (Optional[str]): The name of the service this span belongs to.
        end_time (Optional[str]): The end time of the span.
        span_attributes (Dict[str, str]): Attributes specific to this span (e.g.,
        question, prompts, retrieved documents, llm results).
        resource_attributes (Dict[str, str]): Resource attributes specific to the
        span are derived from the serialized data.
        duration (Optional[int]): The duration of the span in microseconds.
        status_code (Optional[str]): The status code indicating the outcome of the span.
        status_message (Optional[str]): A message providing additional details about the status.
    """

    def __init__(
        self,
        trace_id: UUID,
        span_id: UUID,
        parent_span_id: Any,
        start_time: datetime,
        name: str,
        kind: str,
        span_attributes: Dict[str, str],
        resource_attributes: Dict[str, str],
        service_name: Optional[str] = "LangChain",
        end_time: Optional[str] = None,
        duration: Optional[int] = None,
        status_code: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time = start_time
        self.name = name
        self.kind = kind
        self.service_name = service_name
        self.end_time = end_time
        self.span_attributes = span_attributes
        self.resource_attributes = resource_attributes
        self.duration = duration
        self.status_code = status_code
        self.status_message = status_message

    def update(
        self,
        end_time: datetime,
        span_attributes: Dict[str, str],
        status_code: str,
        status_message: Optional[str] = None,
    ) -> None:
        """Updates the end time, span attributes, status code, and status message of the span."""
        self.end_time = end_time
        self.duration = int((self.end_time - self.start_time).total_seconds() * 1000000)
        self.span_attributes.update(span_attributes)
        self.status_code = status_code
        self.status_message = status_message

    def to_list(self) -> List[Any]:
        """Converts the span data into a list format for uploading."""
        return [
            str(self.trace_id) if self.trace_id is not None else "",
            str(self.span_id) if self.span_id is not None else "",
            str(self.parent_span_id) if self.parent_span_id is not None else "",
            self.start_time,
            self.end_time,
            self.duration,
            self.name,
            self.kind,
            self.service_name,
            self.span_attributes,
            self.resource_attributes,
            self.status_code,
            self.status_message,
        ]
