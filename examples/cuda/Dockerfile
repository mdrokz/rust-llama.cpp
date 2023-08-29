FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    clang \
    cargo \
    nano \
    software-properties-common


RUN apt-get clean && \
rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cd examples/cuda && \
    cargo build --release

CMD ["./examples/cuda/target/release/llama_cuda"]