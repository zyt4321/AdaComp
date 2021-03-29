#!/bin/bash

tc qdisc add dev eth0 root tbf rate ${SPEED}bit burst 10mbit latency 50ms
python3 -u main_new.py --type_node Worker --nb_workers ${WORKER_ID} --ip_PS ${SERVER_IP} --batch_size 128 "$@"