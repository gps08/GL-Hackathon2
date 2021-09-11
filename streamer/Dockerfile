FROM ubuntu:bionic

WORKDIR /app

ENV SCALA_VERSION 2.12
ENV KAFKA_VERSION 2.8.0
ENV KAFKA_HOME /opt/kafka_$SCALA_VERSION-$KAFKA_VERSION
ENV KAFKA_HOST 192.168.0.106
ENV KAFKA_PORT 9092
ENV NUM_PARTITIONS 1

# Install java
RUN apt update && apt install openjdk-8-jre -y

# Install Kafka, Zookeeper and other needed things
RUN apt-get update && \
    apt-get install -y zookeeper wget supervisor dnsutils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    wget -q https://dlcdn.apache.org/kafka/$KAFKA_VERSION/kafka_$SCALA_VERSION-$KAFKA_VERSION.tgz -O /tmp/kafka_$SCALA_VERSION-$KAFKA_VERSION.tgz && \
    tar xfz /tmp/kafka_$SCALA_VERSION-$KAFKA_VERSION.tgz -C /opt && \
    rm /tmp/kafka_$SCALA_VERSION-$KAFKA_VERSION.tgz

COPY . /app

EXPOSE 9092
ENTRYPOINT bash start_kafka.sh
