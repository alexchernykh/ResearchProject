#FROM python:3.8-slim
FROM ubuntu:latest
MAINTAINER fnndsc "dev@babymri.org"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get --yes install git

ADD /detectron/ /detectron
ADD requirements.txt .
ADD /configs /configs
ADD start.sh .

WORKDIR /

RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#RUN export PYTHONPATH='/detectron'

EXPOSE 8080
CMD bash -C start.sh
#CMD["sh", "start.sh"]