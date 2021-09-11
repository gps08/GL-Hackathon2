#!/usr/bin/python3

from time import sleep
from json import dumps
from kafka import KafkaProducer
import sys

BROKER = '0.0.0.0:9092'
TOPIC = 'parking_stats'

if len(sys.argv)>1:
    BROKER = sys.argv[1]

try:
    p = KafkaProducer(bootstrap_servers=BROKER, api_version=(2,8,0),
        value_serializer=lambda x: dumps(x).encode('utf-8'))
except Exception as e:
    print(f"Error connecting to Kafka Server: --> {e}")
    sys.exit(1)

data = open('stream_data.csv', 'r').read().split('\n')
header = data[0].split(',')
data = data[1:]

for i in data:
    message = {}
    tmp = i.split(',')
    for j in range(len(header)):
        message[header[j]] = tmp[j]
    p.send(TOPIC, value=message)
    sleep(1)
