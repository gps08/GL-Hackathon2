#!/usr/bin/python3

from kafka import KafkaConsumer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from json import loads
import sys
from joblib import load
import numpy as np
import pandas as pd

BROKER = '0.0.0.0:9092'
TOPIC = 'parking_stats'

if len(sys.argv)>1:
    BROKER = sys.argv[1]

try:
    consumer = KafkaConsumer(
        TOPIC, api_version=(2,8,0),
        bootstrap_servers=BROKER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='my-group',
        value_deserializer=lambda x: loads(x.decode('utf-8')))
except Exception as e:
    print(f"Error connecting to Kafka Server: {e}")
    sys.exit(1)

model = load('developed.joblib')
encoder = load('encoder.joblib')

def predict(data):
    x = {'row_1': list(data.values())}
    x = pd.DataFrame.from_dict(x, orient='index', columns=list(data.keys()))
    x = encoder.transform(x)
    return model.predict(x)[0]

for message in consumer:
    message = message.value
    print(f'input {message}')
    print(f'prediction output {predict(message)}')
