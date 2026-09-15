FROM python:3.13-slim

WORKDIR /app

# PyMuPDF and friends need a C toolchain to build from source on some
# architectures (e.g. ARM) when a prebuilt wheel isn't available.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

# Mount point for the persistent volume (SQLite, page JSON files, the
# index snapshot). Individual subdirectories are created at startup by
# ApplicationContainer.initialize().
RUN mkdir -p /data

ENV DATA_DIR=/data \
    RAW_DATA_DIR=/data/raw \
    PROCESSED_DATA_DIR=/data/processed \
    INDEX_DIR=/data/indexes \
    METADATA_DIR=/data/metadata

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
