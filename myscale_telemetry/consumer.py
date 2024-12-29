import logging
import time
from queue import Empty, Queue
from threading import Thread
from typing import List, Any

import backoff
from clickhouse_connect.driver.client import Client


class Consumer(Thread):
    """A consumer thread that uploads data to the MyScale vector database.

    This class is responsible for consuming data from a queue and uploading it to the MyScale
    database in batches. It runs as a separate thread to ensure that data is uploaded
    asynchronously without blocking the main application.

    Attributes:
        _log (Logger): The logger for the consumer thread.
        _identifier (int): A unique identifier for the consumer thread.
        _queue (Queue): The queue from which to consume data.
        _client (Client): The MyScale database client for uploading data.
        _upload_interval (float): The interval between uploads in seconds.
        _max_retries (int): The maximum number of retries for uploading data.
        _max_batch_size (int): The maximum batch size for uploading data.
        _database_name (str): The name of the database to use.
        _table_name (str): The name of the table to use.
    """

    _log = logging.getLogger(__name__)
    _identifier: int
    _queue: Queue
    _client: Client
    _upload_interval: float
    _max_retries: int
    _max_batch_size: int
    _database_name: str
    _table_name: str

    def __init__(
        self,
        identifier: int,
        queue: Queue,
        client: Client,
        upload_interval: float,
        max_retries: int,
        max_batch_size: int,
        database_name: str,
        table_name: str,
    ) -> None:
        """Initialize the consumer thread."""
        Thread.__init__(self, daemon=True)
        self._identifier = identifier
        self.running = True
        self._queue = queue
        self._client = client
        self._upload_interval = upload_interval
        self._max_retries = max_retries
        self._max_batch_size = max_batch_size
        self._database_name = database_name
        self._table_name = table_name

        self._log.debug("start creating trace database and table if not exists")
        self._client.command(f"""CREATE DATABASE IF NOT EXISTS {self._database_name}""")
        self._client.command(
            f"""CREATE TABLE IF NOT EXISTS {self._database_name}.{self._table_name}
(
    `TraceId` String CODEC(ZSTD(1)),
    `SpanId` String CODEC(ZSTD(1)),
    `ParentSpanId` String CODEC(ZSTD(1)),
    `StartTime` DateTime64(9) CODEC(Delta(8), ZSTD(1)),
    `EndTime` DateTime64(9) CODEC(Delta(8), ZSTD(1)),
    `Duration` Int64 CODEC(ZSTD(1)),
    `SpanName` LowCardinality(String) CODEC(ZSTD(1)),
    `SpanKind` LowCardinality(String) CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `SpanAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StatusCode` LowCardinality(String) CODEC(ZSTD(1)),
    `StatusMessage` String CODEC(ZSTD(1)),
    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_key mapKeys(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_value mapValues(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_duration Duration TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toDate(StartTime)
ORDER BY (SpanName, toUnixTimestamp(StartTime), TraceId)
SETTINGS index_granularity = 8192"""
        )

    def run(self) -> None:
        """Run the consumer thread."""
        self._log.debug("consumer %s is running...", self._identifier)
        while self.running:
            self.upload()
            time.sleep(self._upload_interval)

        self.upload()
        self._log.debug("consumer %s stopped", self._identifier)

    def stop(self) -> None:
        """Stop the consumer."""
        self._log.debug("stop consumer %s", self._identifier)
        self.running = False

    def get_batch(self) -> List[Any]:
        """Return a batch of items if exists"""
        items = []

        while len(items) < self._max_batch_size:
            try:
                item = self._queue.get_nowait()
                items.append(item)
            except Empty:
                break

        return items

    def upload(self) -> None:
        """Upload a batch of items to MyScale, return whether successful."""

        batch_data = self.get_batch()
        if len(batch_data) == 0:
            return

        try:
            self.upload_batch(batch_data)
        except Exception as e:
            self._log.exception("error uploading data to MyScale: %s", e)
        finally:
            for _ in batch_data:
                self._queue.task_done()

    def upload_batch(self, batch_data: List[Any]) -> None:
        """Upload a batch of items to MyScale with retries."""
        self._log.debug("uploading batch data: %s to MyScale", batch_data)

        @backoff.on_exception(backoff.expo, Exception, max_tries=self._max_retries)
        def insert_with_backoff(batch_data_: List[Any]):
            return self._client.insert(
                table=self._table_name, database=self._database_name, data=batch_data_
            )

        insert_with_backoff(batch_data)
        self._log.debug("successfully uploaded batch of %d items", len(batch_data))
