FROM python:3.7.16-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN  apt update
RUN  pip3 install --upgrade pip
RUN  apt-get --yes install git
RUN  apt-get --yes install wget

ADD /detectron/ /detectron
ADD requirements.txt .
ADD /configs /configs
ADD start.sh .

COPY ./ports.conf /etc/apache2/ports.conf
COPY ./apache.conf /etc/apache2/site-enabled/000-default.conf

#download detectron weights and place them at a specific location
WORKDIR /

RUN  python -m pip install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html

#install dependencies + detectron libs as git source
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

RUN mkdir -p /detectron/output/crossval_full/20190917
RUN wget https://github.com/alexchernykh/ResearchProject/releases/download/modelWeights/model_final_FIRST_INCISION_20190917.pth -O /detectron/output/crossval_full/20190917/model_final_FIRST_INCISION_20190917.pth

EXPOSE 80
CMD bash -C start.sh
