FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

RUN apt-get update && apt-get install -y curl && \
    rm -rf /opt/conda && \
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"

COPY . /workspace/

WORKDIR /workspace/

RUN conda env create -f /workspace/env/minigpt_env.yml

#Activate the environment by default
RUN echo "source activate certifiedgpt" >> ~/.bashrc

# RUN mkdir -p storage \
#     && mkdir -p storage/dataset \
#     && mkdir -p storage/checkpoints/llm_model
#
# ENV DATA_DIR="/storage/dataset"
# ENV VICUNA_DIR="/storage/checkpoints/llm_model"

# Default command
CMD ["bash"]
