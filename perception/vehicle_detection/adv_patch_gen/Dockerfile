# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Builds ultralytics/yolov5:latest image on DockerHub https://hub.docker.com/r/ultralytics/yolov5
# Image is CUDA-optimized for YOLOv5 single/multi-GPU training and inference

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM docker.io/pytorch/pytorch:latest
FROM pytorch/pytorch:latest

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl

# set username inside docker
ARG UNAME=user1
ARG UID=1000

# add user UNAME as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -g root -G sudo -u "$UID" "$UNAME"
# activate user
USER "$UNAME"

ENV PATH="/home/$UNAME/.local/bin:${PATH}"

# Create working directory
RUN rm -rf "/home/$UNAME/app" && mkdir -p "/home/$UNAME/app"
WORKDIR "/home/$UNAME/app"

# Install pip packages
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt albumentations comet gsutil notebook \
    coremltools onnx onnx-simplifier onnxruntime 'openvino-dev>=2022.3'
    # tensorflow tensorflowjs \

COPY adv_patch_gen/requirements.txt "/home/$UNAME/app/adv_patch_gen/requirements.txt"
RUN pip install --no-cache -r "/home/$UNAME/app/adv_patch_gen/requirements.txt"

# Set environment variables
ENV OMP_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cleanup
ENV DEBIAN_FRONTEND teletype

# Copy contents
# (might have issues as not a .git directory)
COPY . "/home/$UNAME/app"
# RUN git clone https://github.com/ultralytics/yolov5 /usr/src/app

# change file ownership to docker user
USER root
RUN chown -R "$UNAME" "/home/$UNAME/app"
USER "$UNAME"

