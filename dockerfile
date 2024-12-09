# FROM pytorch/pytorch:latest
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# COPY /env/minigpt_env.yml /tmp/minigpt_env.yml

COPY . /

RUN apt-get update && apt-get install -y curl && \
    rm -rf /opt/conda && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init

ENV PATH="opt/conda/bin:$PATH"

RUN conda env create -f /env/minigpt_env.yml

#Activate the environment by default
RUN echo "source activate pytorch_env" >> ~/.bashrc

# set the working directory
WORKDIR /

#make the shell interactive
CMD ["bash"]
