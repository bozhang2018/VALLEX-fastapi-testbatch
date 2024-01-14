FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /app

# Add the deadsnakes PPA, update package list, and install Python 3.10
ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to enable adding PPAs
RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA to get access to Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# Update package list again and install Python 3.10
RUN apt-get update && apt-get install -y python3.10
RUN apt-get update && apt-get install -y python3-pip

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y ffmpeg

COPY ./requirements-docker.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy your FastAPI application
COPY ./app /app

CMD ["uvicorn", "main:app", "--reload", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

