from setuptools import setup, find_packages

setup(
    name="myscale_callback",
    version="0.2.0",
    description="A package for MyScale Callback Handler",
    author="Xu Jing",
    author_email="xuj@myscale.com",
    url="https://github.com/myscale/myscale_callback.git",
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
