FROM python:3.11-slim

# Instalamos Chromium y chromedriver
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

WORKDIR /app
COPY /src .
RUN pip install --no-cache-dir -r requirements.txt

# Puerto que usará uvicorn dentro del contenedor
ENV PORT=10000

CMD ["uvicorn", "json2tab_api:app", "--host", "0.0.0.0", "--port", "10000"]