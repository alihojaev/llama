## LaMa (advimman/lama) на Runpod — FastAPI сервис

Репозиторий для деплоя LaMa Inpainting модели (`advimman/lama`) как API-сервис на FastAPI. Модельные веса скачиваются автоматически из Hugging Face при старте контейнера.

### Что внутри
- Dockerfile (CUDA 12.2 runtime + Python 3.8)
- requirements.txt (включая `torch==1.8.0`, `torchvision==0.9.0`)
- app.py (FastAPI, эндпоинт POST `/inpaint`)
- start.sh (скачивает модель Big LaMa и запускает Uvicorn)
- Клонирование `https://github.com/advimman/lama` в `/workspace/lama`

### Примечание по весам модели
Оригинальные ссылки с Yandex Disk устарели. Веса скачиваются из Hugging Face:

`https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip`

Скрипт `start.sh` загрузит архив и развернёт его в `/workspace/lama/big-lama`, если этой папки ещё нет.

---

### Локальная сборка и запуск

Требования: установлен Docker и NVIDIA драйверы/рантайм для GPU.

```bash
docker build -t lama-runpod .
docker run --gpus all -p 7860:7860 lama-runpod
```

Проверка готовности:

```bash
curl http://localhost:7860/health
```

Пример запроса на инпейнтинг:

```bash
curl -X POST http://localhost:7860/inpaint \
  -H 'Content-Type: application/json' \
  -d '{"image":"<base64>","mask":"<base64>"}'
```

Где:
- `image`: base64 изображения (PNG/JPEG)
- `mask`: base64 бинарной маски (белый=255 — область для восстановления)

Ответ:

```json
{ "result": "<base64 PNG>" }
```

---

### Деплой на Runpod (Custom Container)

1. Соберите образ и при необходимости запушьте в свой реестр (Docker Hub/GHCR):
   ```bash
   docker build -t <your-registry>/lama-runpod:latest .
   docker push <your-registry>/lama-runpod:latest
   ```
2. В Runpod создайте новый Pod типа "Custom Container" с GPU runtime.
3. Укажите образ `<your-registry>/lama-runpod:latest`.
4. Убедитесь, что GPU доступен (например, T4/A10/RTX и т.п.).
5. Откройте порт 7860 (контейнер его слушает). В Runpod укажите HTTP порт 7860.
6. После запуска pod, обращайтесь к эндпоинтам:
   - `GET /health`
   - `POST /inpaint`

Контейнер на старте скачает архив `big-lama.zip` из Hugging Face и развернёт веса. После этого API будет доступен на порту `7860`.

---

### Детали реализации

- Базовый образ: `nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04`
- Клон репозитория LaMa: `/workspace/lama`
- Стартовая команда: `CMD ["bash", "/workspace/start.sh"]`
- Запуск инференса внутри API:
  ```bash
  python3 bin/predict.py model.path=/workspace/lama/big-lama \
      indir=/workspace/input/<req_id> outdir=/workspace/output/<req_id>
  ```

Если вы хотите кастомизировать поведение масок/разрешения — смотрите `bin/predict.py` в исходном репозитории `advimman/lama`.


