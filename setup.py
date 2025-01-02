from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="myscale-telemetry",
    version="0.3.2",
    description="Open-source observability for your LLM application.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xu Jing",
    author_email="xuj@myscale.com",
    url="https://github.com/myscale/myscale-telemetry",
    packages=find_packages(),
    install_requires=[
        "backoff>=2.2.1",
        "langchain~=0.3.0",
        "langchain-community~=0.3.0",
        "clickhouse-connect>=0.7",
        "langchain-openai~=0.2.0",
        "tiktoken>=0.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
