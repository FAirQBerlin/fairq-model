FROM python:3.10-slim

RUN pip install --upgrade pip && pip install pipenv

WORKDIR /usr/src/app

COPY . ./

ARG DEV
RUN set -ex && pipenv install $DEV --deploy --system

