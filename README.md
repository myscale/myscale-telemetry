# MyScale Telemetry

The MyScale Telemetry is a powerful tool designed to enhance the observability of LLM applications by capturing trace data from LangChain-based applications and storing it in [MyScaleDB](https://github.com/myscale/MyScaleDB) or ClickHouse. This enables developers to diagnose issues, optimize performance, and gain deeper insights into their applications' behavior.

<p align="center">
<img src="https://github.com/myscale/myscale-telemetry/blob/main/assets/workflow.png?raw=True" width=700 alt="Workflow of MyScale Telemetry">
</p>

## Installation

Install the MyScale Telemetry package using pip:

```bash
pip install myscale-telemetry
```

## Usage

Here is an example of how to use the `MyScaleCallbackHandler` with LangChain:

```python
import os
from operator import itemgetter
from myscale_telemetry.handler import MyScaleCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import MyScale, MyScaleSettings
from langchain_core.runnables import RunnableConfig

# set up the environment variables for OpenAI and MyScale Cloud/MyScaleDB
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["MYSCALE_HOST"] = "YOUR_MYSCALE_HOST"
os.environ["MYSCALE_PORT"] = "YOUR_MYSCALE_PORT"
os.environ["MYSCALE_USERNAME"] = "YOUR_USERNAME"
os.environ["MYSCALE_PASSWORD"] = "YOUR_MYSCALE_PASSWORD"

# for MyScale cloud, you can set index_type="MSTG" for better performance compared to SCANN
vectorstore = MyScale.from_texts(["harrison worked at kensho"], embedding=OpenAIEmbeddings(), config=MyScaleSettings(index_type="SCANN"))
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

# integrate MyScaleCallbackHandler to capture trace data during the chain execution
chain.invoke({"question": "where did harrison work"}, config=RunnableConfig(
    callbacks=[
        MyScaleCallbackHandler()
    ]
))
```

In the default scenario, the callback handler generates a `trace_id` for a single Agent call. However, if you wish to integrate the Trace of the LLM call process with a higher-level caller, you can configure the `RunnableConfig` to pass in metadata with `trace_id` as the key during the call, as in the following example:

```python
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
* `force_count_tokens`: Forces the calculation of LLM token usage, useful when OpenAI LLM streaming is enabled and token usage is not returned (default: False)
* `encoding_name`: Name of the encoding used by tiktoken. This is only relevant if `force_count_tokens` is set to True (default: cl100k_base)

## Observability

To display trace data collected through the MyScale Telemetry from the LLM Application runtime easily and clearly, we also provide a [Grafana Trace Dashboard](https://github.com/myscale/myscale-telemetry/blob/main/dashboard/grafana_myscale_trace_dashboard.json).
The dashboard allows users to monitor the status of the LLM Application which is similar to LangSmith, making it easier to debug and improve its performance.

### Requirements

* [Grafana](https://grafana.com/grafana)
* [Official ClickHouse data source for Grafana](https://grafana.com/grafana/plugins/grafana-clickhouse-datasource/)
* A compatible database instance. MyScale Telemetry supports [MyScaleDB](https://github.com/myscale/MyScaleDB), [MyScale Cloud](https://myscale.com/), and ClickHouse.

### Set up the Trace Dashboard

Once you have Grafana, installed the ClickHouse data source plugin, and have a MyScale cluster with trace data collected through MyScale Telemetry, follow these steps to set up the MyScale Trace Dashboard in Grafana:

1. **Add a new ClickHouse Data Source in Grafana:**

   In the Grafana Data Source settings, add a new ClickHouse Data Source. Enter the Server Address, Server Port, Username, and Password to match those of your MyScale Cloud/MyScaleDB.
   ![Add a data source](https://github.com/myscale/myscale-telemetry/blob/main/assets/add_data_source.png?raw=True)
   ![Configure the data source](https://github.com/myscale/myscale-telemetry/blob/main/assets/config_data_source.png?raw=True)

2. **Import the MyScale Trace Dashboard:**

   Once the ClickHouse Data Source is added, you can import the [MyScale Trace Dashboard](https://github.com/myscale/myscale-telemetry/blob/main/dashboard/grafana_myscale_trace_dashboard.json?raw=True).

   ![Import the MyScale Trace Dashboard](https://github.com/myscale/myscale-telemetry/blob/main/assets/import_dashboard.png?raw=True)

3. **Configure the Dashboard:**

   After importing, select the MyScale Cluster (ClickHouse Data Source Name), the database name, table name, and TraceID of the trace you want to analyze. The dashboard will then display the Traces Table and the Trace Details Panel of the selected trace.

   ![Dashboard Example](https://github.com/myscale/myscale-telemetry/blob/main/assets/dashboard.png?raw=True)

The MyScale Trace Dashboard provides comprehensive insights into the runtime behavior of your LLM applications, similar to LangSmith. It displays critical information that helps in debugging, optimizing, and understanding the performance of your applications.

## Roadmap

* [ ] Support for more LLM frameworks
* [ ] Support LLM tracing directly
* [ ] Extend to end-to-end GenAI system observability

## Acknowledgment

We give special thanks for these open-source projects:

* [LangChain](https://github.com/langchain-ai/langchain): The most popular LLM framework integrated with MyScale Telemetry.
* [OpenTelemetry](https://opentelemetry.io/): The schema of MyScale Telemetry is inspired by this widely used system telemetry toolset.
* [MyScaleDB](https://github.com/myscale/MyScaleDB) and [ClickHouse](https://github.com/ClickHouse/ClickHouse): Data collected by MyScale Telemetry can be stored in either of these databases.
