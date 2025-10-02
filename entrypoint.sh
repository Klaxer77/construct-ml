#!/bin/bash
set -e

# Прогрев моделей PaddleOCR (один раз)
if [ ! -d "/root/.paddleocr" ]; then
  echo "Downloading PaddleOCR models..."
  python3 - <<'EOF'
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang='ru')
EOF
fi

# Запуск uvicorn
exec uvicorn app:app --host 0.0.0.0 --port 9000
