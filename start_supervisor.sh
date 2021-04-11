#!/bin/bash

python3 -u main_new.py --type_node Supervisor --id_worker M000 --ip_PS ${SERVER_IP} --port ${SERVER_PORT} --mid ${MODEL_ID} "$@"