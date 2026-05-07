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

# Precompute deterministic baseline simulations once at build time so they're
# loaded as a static dict on import — removes ~300ms of GIL-bound work from
# AirlineRM.__init__ that was serialising setup under high session concurrency.
RUN uv run python /app/build_baselines.py

# Same idea for the agent-side initial-bookings simulation: precompute the
# per-task booking state and post-bookings rng state, so __init__ becomes a
# dict lookup instead of running tens of thousands of synchronous RNG draws.
RUN uv run python /app/build_initial_bookings.py

EXPOSE 8080
CMD ["uv", "run", "python", "/app/server.py"]
