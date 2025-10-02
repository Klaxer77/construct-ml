FROM python:3.10-slim

# Системные пакеты
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*


RUN python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
WORKDIR /app
COPY . .

# Копируем entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Порт
EXPOSE 9000

# Запуск
ENTRYPOINT ["/entrypoint.sh"]
