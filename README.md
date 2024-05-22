# MyScale Callback

The MyScale Callback Handler is a powerful tool designed to enhance the observability of LLM applications by capturing trace data from LangChain-based applications and storing it in the [MyScale database](https://myscale.com/). This enables developers to diagnose issues, optimize performance, and gain deeper insights into their applications' behavior.

## Installation

Install the MyScale Callback Handler package using pip:

```bash
pip install myscale-callback
```

## Usage

Here is an example of how to use the `MyScaleCallbackHandler` with LangChain:

```python
import os
from operator import itemgetter
from myscale_callback.handler import MyScaleCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import MyScale
from langchain_core.runnables import RunnableConfig

# Set the database configuration through environment variables
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["MYSCALE_HOST"] = "YOUR_MYSCALE_HOST"
os.environ["MYSCALE_PORT"] = "YOUR_MYSCALE_PORT"
os.environ["MYSCALE_USERNAME"] = "YOUR_USERNAME"
os.environ["MYSCALE_PASSWORD"] = "YOUR_MYSCALE_PASSWORD"

vectorstore = MyScale.from_texts(["harrison worked at kensho"], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
model = ChatOpenAI()
template = """Answer the question based only on the following context:
{context}

Question: {question}

"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Set the MyScale callback handler for chain runtime
chain.invoke({"question": "where did harrison work"}, config=RunnableConfig(
    callbacks=[
        MyScaleCallbackHandler()
    ]
))
```

In the default scenario, MyScale Callback generates a `trace_id` for a single Agent call. However, if you wish to integrate the Trace of the LLM call process with a higher-level caller, you can configure the `RunnableConfig` to pass in metadata with `trace_id` as the key during the call, as in the following example:
```
# trace_id obtained from the upper layer, such as request_id of http request
trace_id = "http-request-id-xxx"
chain.invoke({"question": "where did harrison work"}, config=RunnableConfig(
        callbacks=[
            MyScaleCallbackHandler()
        ],
        metadata={"trace_id": trace_id}
    ))
```

## Custom Parameters

When invoking `MyScaleCallbackHandler()`, you can specify several parameters to customize its behavior. If not specified, the default values will be used.

* `myscale_host`: MyScale database host (can also be set via `MYSCALE_HOST` environment variable)             
* `myscale_port`: MyScale database port (can also be set via `MYSCALE_PORT` environment variable)             
* `myscale_username`: MyScale database username (can also be set via `MYSCALE_USERNAME` environment variable) 
* `myscale_password`: MyScale database password (can also be set via `MYSCALE_PASSWORD` environment variable) 
* `threads`: Number of upload threads (default: 1)
* `max_retries`: Maximum number of upload retries (default: 10)
* `max_batch_size`: Maximum upload batch size (default: 1000)
* `max_task_queue_size`: Maximum upload task queue size (default: 10000)
* `upload_interval`: Upload interval in seconds (default: 0.5)
* `database_name`: Name of the trace database (default: "otel")
* `table_name`: Name of the trace table (default: "otel_traces")
