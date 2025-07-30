# Dockerfile Sources
# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
FROM python:3.12-slim

ARG DEV=false

ENV TZ=Europe/Berlin \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    ca-certificates \
    # Install Poetry - keep poetry deps isolated from project deps
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get autoremove -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$PATH:/root/.local/bin"

# Copy only requierments to cache in layer
WORKDIR /usr/src/app

# Copy & Update CA certificates
COPY certificates/INWT-IPA-CA.pem /usr/local/share/ca-certificates/INWT-IPA-CA.crt
RUN update-ca-certificates

# Install dependencies
COPY pyproject.toml poetry.lock* README.md ./

RUN echo "DEV value is: ${DEV}" && \
    # Install dependencies based on DEV_DEPS argument
    if [ "${DEV}" = "true" ]; then \
      poetry install --with dev --no-root --no-interaction --no-ansi; \
    else \
      poetry install --without dev --no-root --no-interaction --no-ansi; \
    fi

# Install package
COPY . ./
RUN poetry install --only-root
