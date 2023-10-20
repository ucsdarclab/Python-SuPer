# Change this to the desired CUDA version
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04   

RUN apt-get update -y 
RUN apt-get install -y gcc curl git libgl1-mesa-glx

#Obtain Micromamba
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
RUN bash Mambaforge-$(uname)-$(uname -m).sh -bfp /usr/local 
RUN mamba install python=3.8

# Build mamba env from resources/environment.yaml
RUN mkdir -p /app
COPY resources/environment.yaml env.yaml 
RUN mamba env create -n super --file env.yaml
SHELL ["mamba","run","-n","super","/bin/bash","-c"]
SHELL ["/bin/bash","-c"]

RUN mamba init
RUN echo 'mamba activate super' >> /root/.bashrc
RUN pip install vedo && rm -rf $(pip cache dir)