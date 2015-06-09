#!/bin/bash

ipengines=$1
if [ -z "$1" ]
then
    ipengines=$(pgrep ipengine|wc -l)
fi

killall ipengine

for i in `seq 1 $ipengines`;
do
    ipengine --ssh="staszek.jastrzebski@104.155.96.215" --file=ipcontroller-engine.json --profile=local &
done  
