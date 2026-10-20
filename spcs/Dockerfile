# Shape AE autoresearch image for Snowpark Container Services.
# Holds the fixed harness only; each experiment's train.py arrives through the stage.
# Build via `uv run spcs_run.py build` (tags the image with the harness hash).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
# torch from PyPI ships its CUDA runtime; the SPCS GPU node provides the driver
RUN pip install "torch==2.9.1" "numpy>=2.2" "pandas>=2.3" "pyarrow>=21" "matplotlib>=3.10" \
        "snowflake-connector-python[pandas]>=3.12"

WORKDIR /app
COPY prepare.py data_sources.py events.py fera.toml ./
COPY tests/ ./tests/
COPY spcs/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
