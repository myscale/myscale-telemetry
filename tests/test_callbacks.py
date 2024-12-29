import time
import logging
import logging.config
from operator import itemgetter
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import MyScale
from myscale_telemetry.handler import MyScaleCallbackHandler

logging.config.fileConfig("logging.conf")


def test_callback_handler():
    """Test the MyScaleCallbackHandler"""
    dotenv.load_dotenv()
    # pylint: disable=no-member
    vectorstore = MyScale.from_texts(
        ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
    )
    # pylint: enable=no-member
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

    test_database = "trace_test"
    test_table = "traces"
    test_question = "where did harrison work"
    callback_handler = MyScaleCallbackHandler(
        database_name=test_database, table_name=test_table
    )

    chain.invoke(
        {"question": test_question}, config=RunnableConfig(callbacks=[callback_handler])
    )

    time.sleep(5)
    trace_id = callback_handler.myscale_client.query(
        f"SELECT TraceId FROM {test_database}.{test_table} "
        f"WHERE SpanAttributes['question'] = '{test_question}' "
        f"AND ParentSpanId = '' order by StartTime DESC limit 1"
    ).result_columns[0][0]

    trace_root = callback_handler.myscale_client.query(
        f"SELECT * FROM {test_database}.{test_table} "
        f"WHERE TraceId = '{trace_id}' AND ParentSpanId = ''"
    )
    logging.info(
        "Callback traces:\n%s", "\n".join(str(d) for d in trace_root.named_results())
    )

    assert (
        callback_handler.myscale_client.query(
            f"SELECT count(*) FROM {test_database}.{test_table} WHERE TraceId = '{trace_id}' "
            f"AND StatusCode = 'STATUS_CODE_SUCCESS'"
        ).result_columns[0][0]
        == 9
    )
