import atexit
import logging
from contextvars import ContextVar
from uuid import uuid4, UUID
from typing import List, Optional, Any, Dict
from queue import Queue
from datetime import datetime
from clickhouse_connect.driver.client import Client

from .consumer import Consumer
from .span_data import SpanData


def create_uuid() -> UUID:
    """Create a new UUID."""
    return uuid4()


class TaskManager:
    """Manages the tasks of uploading data to the MyScale vector database.

     This class is responsible for managing a queue of tasks that upload span data to the MyScale
     database. It uses a pool of consumer threads to process the tasks asynchronously.

    Attributes:
         _log (Logger): The logger for the task manager.
         _consumers (List[Consumer]): The list of consumer threads.
         _client (Client): The MyScale database client.
         _threads (int): The number of consumer threads.
         _max_retries (int): The maximum number of retries for uploading data.
         _max_batch_size (int): The maximum batch size for uploading data.
         _upload_interval (float): The interval between uploads in seconds.
         _queue (Queue): The queue of tasks.
         _database_name (str): The name of the database to use.
         _table_name (str): The name of the table to use.
    """

    _log = logging.getLogger(__name__)
    _consumers: List[Consumer]
    _client: Client
    _threads: int
    _max_retries: int
    _max_batch_size: int
    _upload_interval: float
    _queue: Queue
    _database_name: str
    _table_name: str

    def __init__(
        self,
        client: Client,
        threads: int,
        max_retries: int,
        max_batch_size: int,
        max_task_queue_size: int,
        upload_interval: float,
        database_name: str,
        table_name: str,
    ) -> None:
        """Initializes the TaskManager with the MyScale database client and other parameters.

        Parameters:
            client (Client): The MyScale database client.
            threads (int): The number of consumer threads.
            max_retries (int): The maximum number of retries for uploading data.
            max_batch_size (int): The maximum batch size for uploading data.
            max_task_queue_size (int): The maximum size of the task queue.
            upload_interval (float): The interval between uploads in seconds.
            database_name (str): The name of the database to use.
            table_name (str): The name of the table to use.
        """
        self._client = client
        self._threads = threads
        self._max_retries = max_retries
        self._max_batch_size = max_batch_size
        self._upload_interval = upload_interval
        self._queue = Queue(max_task_queue_size)
        self._database_name = database_name
        self._table_name = table_name
        self.spans = {}
        self._consumers = []
        self.trace_id = ContextVar[UUID]("trace_id", default=None)
        self.__init_consumers()
        atexit.register(self.join)

    def __init_consumers(self) -> None:
        """Initialize the consumer threads."""
        for i in range(self._threads):
            consumer = Consumer(
                identifier=i,
                queue=self._queue,
                client=self._client,
                upload_interval=self._upload_interval,
                max_retries=self._max_retries,
                max_batch_size=self._max_batch_size,
                database_name=self._database_name,
                table_name=self._table_name,
            )
            consumer.start()
            self._consumers.append(consumer)

    def create_trace_id(self) -> UUID:
        """Create a new trace ID."""
        trace_id = create_uuid()
        self.trace_id.set(trace_id)
        return trace_id

    def get_trace_id(self) -> Any:
        """Get the current trace ID."""
        return self.trace_id.get()

    def create_span(
        self,
        trace_id: UUID,
        span_id: UUID,
        parent_span_id: Any,
        start_time: datetime,
        name: str,
        kind: str,
        span_attributes: Dict[str, str],
        resource_attributes: Dict[str, str],
    ) -> None:
        """Create a new span data object."""
        if span_id in self.spans:
            raise ValueError(f"Span with {span_id} id already exists")

        self.spans[span_id] = SpanData(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            start_time=start_time,
            name=name,
            kind=kind,
            span_attributes=span_attributes,
            resource_attributes=resource_attributes,
        )

    def end_span(
        self,
        span_id: UUID,
        end_time: datetime,
        span_attributes: Dict[str, str],
        status_code: str,
        status_message: Optional[str] = None,
        force_count_tokens: bool = False,
    ) -> None:
        """End a span and add its data to the queue."""
        if span_id not in self.spans:
            raise ValueError(f"Span with {span_id} id not exists")

        if force_count_tokens:
            span_attributes["total_tokens"] = str(
                int(span_attributes.get("completion_tokens", 0))
                + int(self.spans[span_id].span_attributes.get("prompt_tokens", 0))
            )

        self.spans[span_id].update(
            end_time, span_attributes, status_code, status_message
        )
        self.__add_span_data(self.spans[span_id].to_list())
        del self.spans[span_id]

    def __add_span_data(self, data: List[Any]) -> None:
        """Add span data to the trace queue, return whether successful."""
        self._log.debug("adding span data: %s", data)
        self._queue.put_nowait(data)

    def flush(self) -> None:
        """Flush all data from the internal queue to MyScale."""
        self._log.debug("flushing queue")
        size = self._queue.qsize()
        self._queue.join()

        self._log.debug(
            "successfully flushed about %s span items.",
            size,
        )

    def join(self) -> None:
        """Join all consumer threads."""
        self._log.debug("joining %s consumer threads", len(self._consumers))
        for consumer in self._consumers:
            consumer.stop()
            try:
                consumer.join()
            except RuntimeError:
                # consumer thread has not started
                pass

            self._log.debug(
                "consumer thread %d joined",
                consumer._identifier,  # pylint: disable=protected-access
            )

    def shutdown(self) -> None:
        """Shut down the TaskManager and flush all data."""
        self._log.debug("shutdown initiated")

        self.flush()
        self.join()

        self._log.debug("shutdown completed")
