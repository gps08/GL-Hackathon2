# Docker Demo/Assignment
https://github.com/gps08/price-predictor contains code which demonstrates docker and docker-compose using the IRIS dataset (https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

## Pre-requisites
- Install `docker`
- Install `docker-compose`

## Running Instructions
- Create a fork of the repo using the `fork` button.
- Clone your fork using `git clone https://www.github.com/gps08/price-predictor.git`
- Build the images using `docker compose build`
- Edit the 3 docker files (producer, streamer and predictor) with your system's IP address
- Spin up the containers using `docker compose up`
- Wait for a few minutes for kafka server to come up and we can see the prediction input and output pairs in the terminal window

## Dataset
https://www.kaggle.com/new-york-city/nyc-parking-tickets?select=Parking_Violations_Issued_-_Fiscal_Year_2017.csv

## Problem
We have used pyspark for model development and prediction along with kafka to stream the data in the dataset. Then we have a kafka constantly that tries to predict the voilation location for parking tickets

## Solution
We have used the Pub Sub design pattern for this project consisting of 3 different docker containers
A. Producer: This reads the dataset csv file, and write the data to a kafka stream
B. Streamer: This is a kafka server container that holds the intermediate data until its read by the consumer
C. Consumer/Predictor: This is the container that applies the pre learned model on the kafka inputs and output the predicion results to stdout
