#!/bin/bash

#python3 -u main_new.py --type_node Worker --id_worker ${WORKER_ID} --ip_PS ${SERVER_IP} --batch_size 128 "$@"
python3 -u main_new.py --type_node PS --id_worker 0 --ip_PS ${SERVER_IP} --port ${SERVER_PORT} --mid ${MODEL_ID} "$@"