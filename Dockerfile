FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Установка зависимостей, необходимых для Python и сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Установка pip для Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Опционально: установить python3 по умолчанию на python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Установка Poetry
ENV POETRY_VERSION=2.1.1
RUN curl -sSL https://install.python-poetry.org | python3.11 -

# Добавляем Poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Обязательно: задаём CUDA_HOME, чтобы flash_attn находил nvcc
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости проекта с помощью Poetry
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root
RUN poetry run pip install --upgrade pip setuptools wheel

# Устанавливаем flash_attn с ключом --no-build-isolation,
# чтобы сборка использовала nvcc из CUDA-образа
RUN poetry run pip install flash_attn --no-build-isolation

# Устанавливаем FlagEmbedding из Git
RUN poetry run pip install git+https://github.com/FlagOpen/FlagEmbedding.git

# Копируем исходный код приложения
COPY . .
COPY inference/* inference/

EXPOSE 8080

CMD ["poetry", "run", "python3", "app.py"]
