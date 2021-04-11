FROM ubuntu:18.04

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y iproute2 stress-ng apt-transport-https vim net-tools \
    ca-certificates curl wget software-properties-common git libgl1-mesa-glx \
    libsm6 libxrender1 libxext-dev
RUN alias python=python3 && ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get install -y python3-pip build-essential python3-dev python3-setuptools
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000
RUN python3 -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple


COPY . .
RUN chmod +x start.sh start_worker.sh start_ps.sh start_supervisor.sh
ENTRYPOINT ./start.sh $0 $@