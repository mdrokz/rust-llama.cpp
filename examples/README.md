> [!IMPORTANT]  
> This was tested with Docker running on Linux & Windows on my Linux PC with RX 6700 XT GPU & my brothers Windows PC with NVIDIA RTX 2060 SUPER. <br>If you can test it on other GPUs & Platforms, please update this `README.md` with a PR!<br>

## Supported / Tested

- AMD RX 6700 XT / Fedora 37
- NVIDIA RTX 2060 Super / Windows 11 Docker

# Examples

There are 3 examples basic, cuda and opencl each of them have their own Dockerfile except the basic example.

# basic

A simple example that runs inference on the default options:

```
cargo run --release
```

# cuda

A example to use nvidia GPU's with the cuda feature:

You can also set these env variables for cuda

`LLAMA_CUDA_DMMV_X=32 LLAMA_CUDA_DMMV_Y=1 LLAMA_CUDA_KQUANTS_ITER=2`

firstly build the image from the root of the repository

```
docker build -f examples/cuda/Dockerfile . -t llama_cuda
```

then you can run it:

### linux

```
docker run --device=/dev/dri:/dev/dri --volume=<your directory that contains the models>
:/models llama_cuda
```

### windows

```
docker run --volume=<your directory that contains the models>:/models --gpus all llama_cuda
```


# opencl

A example to run CLBlast supported GPUs:

firstly build the image from the root of the repository

```
docker build -f examples/opencl/Dockerfile . -t llama_opencl
```

then you can run it:

### linux

```
docker run --device=/dev/dri:/dev/dri --volume=<your directory that contains the models>
:/models llama_opencl
```