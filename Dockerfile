FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl build-essential && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.1.1
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

#RUN pip install flash_attn
#RUN pip install git+https://github.com/FlagOpen/FlagEmbedding.git

COPY . .
COPY inference/* inference/

EXPOSE 8080

CMD ["python", "app.py"]
