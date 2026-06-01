# Dockerfile for neural network, protein dynamics simulations will be done with a gromacs image
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY lib ./lib

