FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install transformers
RUN pip install --upgrade datasets
RUN pip install pandas
RUN pip install argparse
RUN pip install 'accelerate>=0.26.0'
