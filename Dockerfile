# A fixed environment for the local pipeline and the tests: the same Python version and the
# same install command as the CI workflow. The Databricks jobs do not use this image; they
# install the wheel on serverless compute (see databricks.yml).
#
#   docker build -t grid-carbon-intensity-mlops .
#   docker run --rm grid-carbon-intensity-mlops                  # lint and tests
#   docker run --rm grid-carbon-intensity-mlops make run-local   # the local pipeline
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime LightGBM loads; make runs the same tasks as CI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# What is copied is limited by .dockerignore: no Git history, no local data stores, no
# credentials, and none of the untracked local folders.
COPY . .
RUN pip install -e ".[dev]"

# Run as an ordinary user; the pipeline writes its tables and MLflow store under /app.
RUN useradd --create-home runner && chown -R runner /app
USER runner

CMD ["make", "test"]
