FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install malaya tqdm tensorflow==1.15

COPY ./app /app
COPY ./start.sh /app
COPY ./load_model.py /app