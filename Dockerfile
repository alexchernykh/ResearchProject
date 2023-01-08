FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get --yes install git \
  && apt-get --yes install wget



ADD /detectron/ /detectron
ADD requirements.txt .
ADD /configs /configs
ADD start.sh .

COPY ./ports.conf /etc/apache2/ports.conf
COPY ./apache.conf /etc/apache2/site-enabled/000-default.conf

#download detectron weights and place them at a specific location
WORKDIR /
RUN mkdir -p /detectron/output/crossval_full/20190917
RUN wget https://github.com/alexchernykh/ResearchProject/releases/download/modelWeights/model_final_FIRST_INCISION_20190917.pth -O /detectron/output/crossval_full/20190917/model_final_FIRST_INCISION_20190917.pth

#install dependencies + detectron libs as git source
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html


EXPOSE 80
CMD bash -C start.sh
#CMD["sh", "start.sh"]
