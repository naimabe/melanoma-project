FROM python:3.10.8-slim
ENV APP_HOME /app

COPY. ./
RUN pip install --upgrade pip
RUN install -r requirements.txt
RUN pip install -e
RUN apt update
RUN apt install direnv
