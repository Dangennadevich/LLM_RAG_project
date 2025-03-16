# Базовый образ Python
FROM python:3.11-slim

# Установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Установка Poetry (укажите нужную версию, например, 1.4.0)
ENV POETRY_VERSION=2.1.1
RUN curl -sSL https://install.python-poetry.org | python3 -

# Добавление Poetry в PATH
ENV PATH="/root/.local/bin:$PATH"

# Задание рабочей директории
WORKDIR /app

# Копирование файлов для установки зависимостей
COPY pyproject.toml poetry.lock ./

# Отключение создания виртуального окружения и установка зависимостей
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

#RUN pip install flash_attn

# Копирование исходного кода приложения
COPY . .
COPY inference/* inference/

# Открытие порта, который использует ваше приложение (при необходимости)
EXPOSE 8080

# Запуск приложения (замените "app.py" на основной модуль вашего сервиса)
CMD ["python", "app.py"]
