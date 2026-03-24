FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y \
    software-properties-common \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
RUN uv venv --python 3.11

COPY . /app
RUN uv pip install -r /app/requirements.txt

EXPOSE 8080
CMD ["uv", "run", "python", "/app/server.py"]
