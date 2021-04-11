#!/bin/bash

#tc qdisc del dev eth0 root
#tc qdisc add dev eth0 root handle 1:  htb default 11
#tc class add dev eth0 parent 1: classid 1:11 htb rate ${SPEED}bit ceil ${SPEED}bit
python3 -u main_new.py --type_node Worker --id_worker W${WORKER_ID} --ip_PS ${SERVER_IP} --port ${SERVER_PORT} "$@"