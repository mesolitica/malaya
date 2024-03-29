FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN pip3 install playwright
RUN pip3 install Pillow
RUN playwright install-deps
RUN playwright install

COPY ./app /app

ENV PORT=9091
ENTRYPOINT ["/start-reload.sh"]