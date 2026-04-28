FROM python:3.13-slim

WORKDIR /app

# Dependências do sistema necessárias para ChromaDB e HuggingFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python primeiro (cache de layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte
COPY bot/       bot/
COPY config/    config/
COPY core/      core/
COPY utils/     utils/
COPY main.py    .

# Copia o índice ChromaDB já processado (38MB — não precisamos dos PDFs originais)
COPY storage/   storage/

# Variáveis de ambiente obrigatórias (fornecidas pelo Railway em runtime)
ENV TELEGRAM_TOKEN=""
ENV GOOGLE_API_KEY=""
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "main.py"]
