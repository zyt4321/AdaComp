#!/bin/bash

type_list=('A' 'B' 'C' 'D')
speed_list=('1m' '1m' '256k' '256k')
cpu_list=('1' '0.25' '1' '0.25')
for idx in $(seq 0 "`expr ${#type_list[@]} - 1`")
do
  for i in $(seq 1 3)
  do
    ID=`echo ${i} | awk '{printf("%02d",$0)}'`
    docker run --cap-add=NET_ADMIN --cpus ${cpu_list[$idx]} -e PYTHONUNBUFFERED=1 -e SPEED=${speed_list[$idx]} -e WORKER_ID=W${type_list[$idx]}${ID} -e SERVER_IP=172.17.0.1 comp &
  done
done
