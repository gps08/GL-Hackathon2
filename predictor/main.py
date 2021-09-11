#!/usr/bin/python3

from kafka import KafkaConsumer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from json import loads
import sys
from joblib import load
import numpy as np

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

def predict(data):
    x = np.array([data.values()])
    # do one hot encoding
    cat_ix = ['Plate ID', 'Registration State', 'Plate Type', 'Issue Date',
       'Violation Code', 'Issuing Agency', 'Violation Precinct',
       'Issuer Precinct', 'Violation Time', 'Date First Observed',
       'Law Section', 'Sub Division', 'Vehicle Year', 'Feet From Curb']
    ct = ColumnTransformer([('o', OneHotEncoder(), cat_ix)], remainder='passthrough')
    x = ct.fit_transform(x)
    print(model.predict(x))

for message in consumer:
    message = message.value
    print(f'input {message}')
    predict(message)
