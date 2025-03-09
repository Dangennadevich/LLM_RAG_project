# Проект по созданию QA-LLM-бота на основе RAG системы для ответов на вопросы по научным статьям с arxiv.org

**convert_pdf2txt_example.ipynb**: Сравнение алгоритмов конвертации, а так же пример обработки PDF и работы с S3

**utils.py**: Функции для работы системы и все что не хочится хранить в основном блоке кода

Возможно понадобится для твоего окружения:

<code> pip install git+https://github.com/huggingface/transformers accelerate </code>

<code> pip install minio load-dotenv pdfplumber qwen-vl-utils[decord]==0.0.8 </code>