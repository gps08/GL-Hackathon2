#!/bin/bash

# Configure host and port for kafka
echo "auto.create.topics.enable=true" >> $KAFKA_HOME/config/server.properties
echo "advertised.host.name=$KAFKA_HOST" >> $KAFKA_HOME/config/server.properties
echo "advertised.port=$KAFKA_PORT" >> $KAFKA_HOME/config/server.properties
sed -r -i "s/(num.partitions)=(.*)/\1=$NUM_PARTITIONS/g" $KAFKA_HOME/config/server.properties

# Start kafka server
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties &
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties