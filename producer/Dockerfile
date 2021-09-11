FROM ubuntu:bionic

ENV PYTHONUNBUFFERED 1
ENV DISPLAY=:99

WORKDIR /app

# Install python
RUN apt update && apt install python3 python3-pip -y

# Install requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT python3 main.py 192.168.0.106:9092