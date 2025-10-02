## Требования к окружению

На машине должны быть установлены:

- **NGC CLI** (для работы с образами NVIDIA)
- **Docker 28.0.0**
- **nvidia-container-toolkit** (для GPU в Docker)
- **CUDA Toolkit 12.8**
- **Anaconda 25.1.1** (для локальных Python-окружений)
- **PyTorch 2.5.1 (Python 3.11)**
- **TensorFlow-GPU 2.4.1 (Python 3.9)**
- **nvidia-open 570.86** (драйвер NVIDIA)

Проверить установку:

`docker --version nvidia-smi nvcc --version conda --version`

## Архитектура

1. **OCR-сервис (PaddleOCR, GPU1)**  
    Принимает изображение → распознаёт текст.  
    Запускается в Docker-контейнере.
2. **LLM-сервис (vLLM + Qwen2-7B AWQ, GPU0)**  
    Принимает текст → выдаёт структурированный JSON.  
	    Запускается в Docker-контейнере.
3. **FastAPI-приложение (Python 3.11)**  
    API-слой, который связывает OCR и LLM.

## Запуск сервисов

### 1. Запуск LLM (GPU0)

	`docker run -d \   --gpus all \   -e CUDA_VISIBLE_DEVICES=0 \   --name llm \   -p 8000:8000 \   -v /home/ubuntu/.cache:/root/.cache \   vllm/vllm-openai:v0.5.4 \   --model Qwen/Qwen2-7B-Instruct-AWQ \   --quantization awq \   --gpu-memory-utilization 1 \   --max-model-len 4096`

Проверка:

`curl http://<ip>:8000/v1/models`

### 2. Запустить OCR (CPU)

Внутри папки `ocr_llm_app/`:

Запустить `docker compose up -d --build`

OCR API будет доступен по адресу:

`http://<ip>:9000/docs`

### 3. Проверка работы

OCR:

`curl -X POST -F "file=@test.png" http://<ip>:9000/ocrWithLlm`

## Стек

### Инфраструктура

- Ubuntu 24.04
    
- Docker 28.0.0
    
- nvidia-container-toolkit
    
- CUDA Toolkit 12.8
    
- NVIDIA драйвер `nvidia-open 570.86`
    
- 1× Tesla T4
    

### OCR

- PaddleOCR
    
- PaddlePaddle-GPU
    
- Pillow, NumPy
    
- FastAPI + Uvicorn
    

### LLM

- vLLM (контейнер `vllm/vllm-openai:v0.5.4`)
    
- Qwen2-7B-Instruct-AWQ (AWQ квантовка)
    
- OpenAI Python SDK