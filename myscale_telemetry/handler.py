import logging
import os
import json
import datetime
from typing import Any, Dict, List, Union, Optional, Sequence, cast
from uuid import UUID
import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.document import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    FunctionMessage,
)
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_community.callbacks.utils import flatten_dict
from .task_manager import TaskManager

STATUS_SUCCESS = "STATUS_CODE_SUCCESS"
STATUS_ERROR = "STATUS_CODE_ERROR"


def get_timestamp() -> datetime.datetime:
    """Return the current UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc)


def _extract_prompt_templates(serialized) -> Dict[str, str]:
    """Extract prompt templates from serialized data."""
    prompts_dict = {}
    flat_dict = flatten_dict(serialized)

    for i, message in enumerate(flat_dict.get("kwargs_messages", [])):
        prompt_template = (
            message.get("kwargs", {})
            .get("prompt", {})
            .get("kwargs", {})
            .get("template")
        )
        if isinstance(prompt_template, str):
            prompts_dict[f"prompts.{i}.template"] = prompt_template

    return prompts_dict


def _convert_message_to_dict(message: BaseMessage, prefix_key: str) -> Dict[str, str]:
    """Convert a message to a dictionary with a specified prefix key."""
    role_mapping = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system",
        ToolMessage: "tool",
        FunctionMessage: "function",
    }

    role = None

    for msg_type, mapped_role in role_mapping.items():
        if isinstance(message, msg_type):
            role = mapped_role
            break

    if role is None and isinstance(message, ChatMessage):
        role = message.role

    if role is not None:
        return {
            f"{prefix_key}role": role,
            f"{prefix_key}content": cast(str, message.content),
        }

    return {}


def _extract_resource_attributes(
    metadata: Dict[str, Any], serialized: Dict[str, Any]
) -> Dict[str, str]:
    """Extract resource attributes from serialized data."""
    resource_attributes: Dict[str, str] = {}

    flat_dict = flatten_dict(serialized) if serialized else {}
    flat_dict.update(metadata)
    for resource_key, resource_val in flat_dict.items():
        resource_attributes.update({resource_key: _serialize_value(resource_val)})

    return resource_attributes


def _serialize_value(value: Any) -> str:
    """Serialize the given value to a string."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return f"<unserializable object: {type(value).__name__}>"


def _extract_span_attributes(data: Any, **kwargs: Any) -> Dict[str, str]:
    """Extract span attributes from the given data, converting serializable objects to strings."""

    span_attributes: Dict[str, str] = {}

    if isinstance(data, dict):
        for attributes_key, attributes_val in data.items():
            if isinstance(attributes_key, str):
                span_attributes[attributes_key] = _serialize_value(attributes_val)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, BaseMessage):
                span_attributes.update(_convert_message_to_dict(item, f"prompts.{i}."))
            elif isinstance(item, Document):
                span_attributes[f"documents.{i}.content"] = item.page_content
            else:
                span_attributes[f"items.{i}.content"] = _serialize_value(item)

    params = kwargs.get("invocation_params", {})
    if params:
        model = (
            params.get("model") or params.get("model_name") or params.get("model_id")
        )
        chat_type = params.get("_type")
        temperature = str(params.get("temperature"))
        span_attributes.update(
            {"model": model, "chat_type": chat_type, "temperature": temperature}
        )

    return span_attributes


def _get_langchain_run_name(serialized: Optional[Dict[str, Any]], **kwargs: Any) -> str:
    """Retrieve the name of a serialized LangChain runnable."""
    if serialized:
        name = serialized.get("name", serialized.get("id", ["Unknown"])[-1])
    else:
        if "name" in kwargs and kwargs["name"]:
            name = kwargs["name"]
        else:
            name = "Unknown"

    return name


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class MyScaleCallbackHandler(BaseCallbackHandler):
    """Callback Handler for MyScale.

    Parameters:
        myscale_host (Optional[str]): The hostname of the MyScale database.
        myscale_port (Optional[int]): The port of the MyScale database.
        myscale_username (Optional[str]): The username for the MyScale database.
        myscale_password (Optional[str]): The password for the MyScale database.
        threads (int): The number of threads for uploading data to MyScale.
        max_retries (int): The maximum number of retries for uploading data to MyScale.
        max_batch_size (int): The maximum batch size for uploading data to MyScale.
        max_task_queue_size (int): The maximum size of the task queue.
        upload_interval (float): The interval between uploads in seconds.
        database_name (str): The name of the database to use.
        table_name (str): The name of the table to use.
        force_count_tokens (bool): Forces the calculation of LLM token usage,
                                   useful when OpenAI LLM streaming is enabled
                                   and token usage is not returned.
        encoding_name (str): The name of the encoding used by tiktoken.
                             This is only relevant if `force_count_tokens` is set to True.

    This handler utilizes callback methods to extract various elements such as
    questions, retrieved documents, prompts, and messages from each callback
    function, and subsequently uploads this data to the MyScale vector database
    for monitoring and evaluating the performance of the LLM application.
    """

    def __init__(
        self,
        myscale_host: Optional[str] = None,
        myscale_port: Optional[int] = None,
        myscale_username: Optional[str] = None,
        myscale_password: Optional[str] = None,
        threads: int = 1,
        max_retries: int = 10,
        max_batch_size: int = 1000,
        max_task_queue_size: int = 10000,
        upload_interval: float = 0.5,
        database_name: str = "otel",
        table_name: str = "otel_traces",
        force_count_tokens: bool = False,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Set up the MyScale client and the TaskManager,
        which is responsible for uploading data to the MyScale vector database."""
        try:
            from clickhouse_connect import get_client
        except ImportError as exc:
            raise ImportError(
                "Could not import clickhouse connect python package. "
                "Please install it with `pip install clickhouse-connect`."
            ) from exc

        self._log = logging.getLogger(__name__)
        self.myscale_host = myscale_host or os.getenv("MYSCALE_HOST")
        self.myscale_port = myscale_port or int(os.getenv("MYSCALE_PORT", "443"))
        self.myscale_username = myscale_username or os.getenv("MYSCALE_USERNAME")
        self.myscale_password = myscale_password or os.getenv("MYSCALE_PASSWORD")
        self.myscale_client = get_client(
            host=self.myscale_host,
            port=self.myscale_port,
            username=self.myscale_username,
            password=self.myscale_password,
        )

        self.force_count_tokens = force_count_tokens
        self.encoding_name = encoding_name

        self._task_manager = TaskManager(
            client=self.myscale_client,
            threads=threads,
            max_retries=max_retries,
            max_batch_size=max_batch_size,
            max_task_queue_size=max_task_queue_size,
            upload_interval=upload_interval,
            database_name=database_name,
            table_name=table_name,
        )

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        self._log.debug(
            "on chain start run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            if metadata and metadata.get("trace_id"):
                trace_id = metadata["trace_id"]
            else:
                if parent_run_id is None:
                    trace_id = self._task_manager.create_trace_id()
                else:
                    trace_id = self._task_manager.get_trace_id()

            name = _get_langchain_run_name(serialized, **kwargs)

            span_attributes = {}
            if isinstance(inputs, str):
                span_attributes["input"] = inputs
            else:
                span_attributes.update(_extract_span_attributes(inputs, **kwargs))

            if name == "ChatPromptTemplate" and serialized:
                span_attributes.update(_extract_prompt_templates(serialized))

            self._task_manager.create_span(
                trace_id=trace_id,
                span_id=run_id,
                parent_span_id=parent_run_id,
                start_time=get_timestamp(),
                name=name,
                kind="chain",
                span_attributes=span_attributes,
                resource_attributes=_extract_resource_attributes(metadata, serialized),
            )

        except Exception as e:
            self._log.exception("An error occurred in on_chain_start: %s", e)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        self._log.debug(
            "on chain end run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            span_attributes = {}
            if isinstance(outputs, str):
                span_attributes["output"] = outputs
            else:
                span_attributes.update(_extract_span_attributes(outputs, **kwargs))

            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes=span_attributes,
                status_code=STATUS_SUCCESS,
                status_message="",
            )

        except Exception as e:
            self._log.exception("An error occurred in on_chain_end: %s", e)

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        self._log.debug(
            "on llm start run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            if metadata and metadata.get("trace_id"):
                trace_id = metadata["trace_id"]
            else:
                trace_id = self._task_manager.get_trace_id()

            span_attributes = _extract_span_attributes(prompts, **kwargs)
            if self.force_count_tokens:
                prompt_tokens = 0
                for i in range(len(prompts)):
                    content_key = "prompts." + str(i) + ".content"
                    if content_key in span_attributes:
                        prompt_tokens += num_tokens_from_string(
                            span_attributes[content_key], self.encoding_name
                        )

                span_attributes["prompt_tokens"] = str(prompt_tokens)

            self._task_manager.create_span(
                trace_id=trace_id,
                span_id=run_id,
                parent_span_id=parent_run_id,
                start_time=get_timestamp(),
                name=_get_langchain_run_name(serialized, **kwargs),
                kind="llm",
                span_attributes=span_attributes,
                resource_attributes=_extract_resource_attributes(metadata, serialized),
            )
        except Exception as e:
            self._log.exception("An error occurred in on_llm_start: %s", e)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""
        self._log.debug(
            "on llm end run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            span_attributes: Dict[str, str] = {}
            for i, generation in enumerate(response.generations):
                generation = generation[0]
                prefix_key = "completions." + str(i) + "."
                if isinstance(generation, ChatGeneration):
                    span_attributes.update(
                        _convert_message_to_dict(generation.message, prefix_key)
                    )
                else:
                    span_attributes[f"{prefix_key}content"] = generation.text

            if self.force_count_tokens:
                completion_tokens = 0
                for i in range(len(response.generations)):
                    content_key = "completions." + str(i) + ".content"
                    if content_key in span_attributes:
                        completion_tokens += num_tokens_from_string(
                            span_attributes[content_key], self.encoding_name
                        )

                span_attributes["completion_tokens"] = str(completion_tokens)
            else:
                if response.llm_output is not None and isinstance(
                    response.llm_output, Dict
                ):
                    token_usage = response.llm_output["token_usage"]
                    if token_usage is not None:
                        span_attributes["prompt_tokens"] = str(
                            token_usage["prompt_tokens"]
                        )
                        span_attributes["total_tokens"] = str(
                            token_usage["total_tokens"]
                        )
                        span_attributes["completion_tokens"] = str(
                            token_usage["completion_tokens"]
                        )

            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes=span_attributes,
                status_code=STATUS_SUCCESS,
                status_message="",
                force_count_tokens=self.force_count_tokens,
            )

        except Exception as e:
            self._log.exception("An error occurred in on_llm_end: %s", e)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        self._log.debug(
            "on chat model start run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            if metadata and metadata.get("trace_id"):
                trace_id = metadata["trace_id"]
            else:
                trace_id = self._task_manager.get_trace_id()

            flattened_messages = [item for sublist in messages for item in sublist]
            span_attributes = _extract_span_attributes(flattened_messages, **kwargs)
            if self.force_count_tokens:
                prompt_tokens = 0
                for i in range(len(flattened_messages)):
                    content_key = "prompts." + str(i) + ".content"
                    if content_key in span_attributes:
                        prompt_tokens += num_tokens_from_string(
                            span_attributes[content_key], self.encoding_name
                        )

                span_attributes["prompt_tokens"] = str(prompt_tokens)

            self._task_manager.create_span(
                trace_id=trace_id,
                span_id=run_id,
                parent_span_id=parent_run_id,
                start_time=get_timestamp(),
                name=_get_langchain_run_name(serialized, **kwargs),
                kind="llm",
                span_attributes=span_attributes,
                resource_attributes=_extract_resource_attributes(metadata, serialized),
            )
        except Exception as e:
            self._log.exception("An error occurred in on_chat_model_start: %s", e)

    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        self._log.debug(
            "on retriever start run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            if metadata and metadata.get("trace_id"):
                trace_id = metadata["trace_id"]
            else:
                trace_id = self._task_manager.get_trace_id()

            self._task_manager.create_span(
                trace_id=trace_id,
                span_id=run_id,
                parent_span_id=parent_run_id,
                start_time=get_timestamp(),
                name=_get_langchain_run_name(serialized, **kwargs),
                kind="retriever",
                span_attributes={"query": query},
                resource_attributes=_extract_resource_attributes(metadata, serialized),
            )

        except Exception as e:
            self._log.exception("An error occurred in on_retriever_start: %s", e)

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        self._log.debug(
            "on retriever end run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            span_attributes: Dict[str, str] = {}
            for i, document in enumerate(documents):
                span_attributes[f"documents.{i}.content"] = document.page_content

            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes=span_attributes,
                status_code=STATUS_SUCCESS,
                status_message="",
            )

        except Exception as e:
            self._log.exception("An error occurred in on_chain_end: %s", e)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        self._log.debug(
            "on tool start run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            if metadata and metadata.get("trace_id"):
                trace_id = metadata["trace_id"]
            else:
                trace_id = self._task_manager.get_trace_id()

            span_attributes = {}
            if isinstance(input_str, str):
                span_attributes["input"] = input_str
            else:
                span_attributes.update(_extract_span_attributes(input_str, **kwargs))

            self._task_manager.create_span(
                trace_id=trace_id,
                span_id=run_id,
                parent_span_id=parent_run_id,
                start_time=get_timestamp(),
                name=_get_langchain_run_name(serialized, **kwargs),
                kind="tool",
                span_attributes=span_attributes,
                resource_attributes=_extract_resource_attributes(metadata, serialized),
            )

        except Exception as e:
            self._log.exception("An error occurred in on_tool_start: %s", e)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        self._log.debug(
            "on tool end run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            span_attributes = {}
            if isinstance(output, str):
                span_attributes["output"] = output
            else:
                span_attributes.update(_extract_span_attributes(output, **kwargs))
            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes=span_attributes,
                status_code=STATUS_SUCCESS,
                status_message="",
            )
        except Exception as e:
            self._log.exception("An error occurred in on_tool_end: %s", e)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        self._log.debug(
            "on tool error run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes={},
                status_code=STATUS_ERROR,
                status_message=f"An error occurred in a tool: {str(error)}",
            )
        except Exception as e:
            self._log.exception("An error occurred in on_tool_error: %s", e)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        self._log.debug(
            "on chain error run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes={},
                status_code=STATUS_ERROR,
                status_message=f"An error occurred in a chain: {str(error)}",
            )
        except Exception as e:
            self._log.exception("An error occurred in on_chain_error: %s", e)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors."""
        self._log.debug(
            "on retriever error run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes={},
                status_code=STATUS_ERROR,
                status_message=f"An error occurred in a retriever: {str(error)}",
            )
        except Exception as e:
            self._log.exception("An error occurred in on_retriever_error: %s", e)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        self._log.debug(
            "on llm error run_id: %s parent_run_id: %s", run_id, parent_run_id
        )
        try:
            self._task_manager.end_span(
                span_id=run_id,
                end_time=get_timestamp(),
                span_attributes={},
                status_code=STATUS_ERROR,
                status_message=f"An error occurred in a llm: {str(error)}",
            )
        except Exception as e:
            self._log.exception("An error occurred in on_llm_error: %s", e)
