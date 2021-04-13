#!/usr/bin/env bash
##!/bin/bash
#
##python3 -u main_new.py --type_node Worker --id_worker ${WORKER_ID} --ip_PS ${SERVER_IP} --batch_size 128 "$@"
#trap "pkill -SIGINT -f main_new.py" 15
#trap "mkdir hahahahah" 15
#python3 -u main_new.py --type_node PS --id_worker 0 --ip_PS ${SERVER_IP} --port ${SERVER_PORT} --mid ${MODEL_ID}
set -x

pid=0

# SIGTERM-handler
term_handler() {
  if [ $pid -ne 0 ]; then
    kill -SIGINT "$pid"
    wait "$pid"
  fi
  exit 143; # 128 + 15 -- SIGTERM
}
# setup handlers
# on callback, kill the last background process, which is `tail -f /dev/null` and execute the specified handler
trap 'kill ${!}; term_handler' SIGTERM

# run application
python3 -u main_new.py --type_node PS --id_worker 0 --ip_PS ${SERVER_IP} --port ${SERVER_PORT} --mid ${MODEL_ID} &
pid="$!"

# wait forever
while true
do
  tail -f /dev/null & wait ${!}
done
