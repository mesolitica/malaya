FROM python:3.7.5-stretch AS base

RUN pip install Flask gunicorn eventlet gevent
RUN pip install malaya tqdm tensorflow==1.15
RUN pip install uwsgi

WORKDIR /app

COPY . /app