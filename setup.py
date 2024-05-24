from setuptools import setup, find_packages

setup(
    name="myscale-telemetry",
    version="0.1.0",
    description="Open-source observability for your LLM application.",
    author="Xu Jing",
    author_email="xuj@myscale.com",
    url="https://github.com/myscale/myscale-telemetry",
    packages=find_packages(),
    install_requires=[
        "backoff>=2.2.1",
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "clickhouse-connect>=0.7.8"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
