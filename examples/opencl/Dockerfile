FROM cebxan/amdgpu-opencl

# install common dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    clang \
    cargo \
    nano \
    clinfo \
    software-properties-common \
    opencl-headers \ 
    ocl-icd-libopencl1 \ 
    ocl-icd-opencl-dev


# install clblast
RUN add-apt-repository ppa:cnugteren/clblast && \
    apt-get update && \
    apt-get install -y libclblast-dev

WORKDIR /app

COPY . .

RUN cd examples/opencl && \
    cargo build --release

CMD ["./examples/opencl/target/release/llama_opencl"]