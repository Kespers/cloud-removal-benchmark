FROM python:3.7-slim

# Installazione dipendenze di sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Installazione librerie Python richieste dal progetto
RUN pip install --upgrade pip
RUN pip install tensorflow==1.15.0 keras==2.2.4 numpy==1.17.0 scipy==1.2.1 rasterio pydot h5py==2.10.0 protobuf==3.20.0 matplotlib

WORKDIR /app    