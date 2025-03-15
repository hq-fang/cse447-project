FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git

RUN pip install --upgrade pip
RUN pip install torch
RUN pip install transformers
RUN pip install --upgrade datasets
RUN pip install pandas
RUN pip install argparse
RUN pip install 'accelerate>=0.26.0'
RUN pip install sentencepiece
RUN pip install bitsandbytes
RUN pip install pynvml==11.5.0
RUN pip install flash-attn==2.5.7
