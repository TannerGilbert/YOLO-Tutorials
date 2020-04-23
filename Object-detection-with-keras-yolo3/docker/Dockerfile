FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y git wget

RUN apt-get update --fix-missing

RUN apt-get install -y python3 && apt-get install -y python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pil python3-lxml python3-tk && \
    pip3 install matplotlib && \
    pip3 install Pillow && \
    pip3 install pandas && \
    pip3 install tensorflow==1.14.0 && \
    pip3 install keras==2.1.5 

RUN git clone https://github.com/qqwweee/keras-yolo3

RUN cd keras-yolo3