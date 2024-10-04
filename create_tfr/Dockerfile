FROM python:3.7.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y procps

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ausslfr_create_tfr.py"]