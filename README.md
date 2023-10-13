# rust_llama.cpp
[![Docs](https://docs.rs/llama_cpp_rs/badge.svg)](https://docs.rs/llama_cpp_rs)
[![Crates.io](https://img.shields.io/crates/v/llama_cpp_rs.svg?maxAge=2592000)](https://crates.io/crates/llama_cpp_rs)

[LLama.cpp](https://github.com/ggerganov/llama.cpp) rust bindings.

The rust bindings are mostly based on https://github.com/go-skynet/go-llama.cpp/

## Building Locally

Note: This repository uses git submodules to keep track of [LLama.cpp](https://github.com/ggerganov/llama.cpp).

Clone the repository locally:

```bash
git clone --recurse-submodules https://github.com/mdrokz/rust-llama.cpp
```

```bash
cargo build
```

## Usage

```toml
[dependencies]
llama_cpp_rs = "0.3.0"
```

```rs
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
    let model_options = ModelOptions::default();

    let llama = LLama::new(
        "../wizard-vicuna-13B.ggmlv3.q4_0.bin".into(),
        &model_options,
    )
    .unwrap();

    let predict_options = PredictOptions {
        token_callback: Some(Box::new(|token| {
            println!("token1: {}", token);

            true
        })),
        ..Default::default()
    };

    llama
        .predict(
            "what are the national animals of india".into(),
             predict_options,
        )
        .unwrap();
}

```

## Examples 

The examples contain dockerfiles to run them

see [examples](https://github.com/mdrokz/rust-llama.cpp/blob/master/examples/README.md)

## TODO

- [x] Implement support for cublas,openBLAS & OpenCL [#7](https://github.com/mdrokz/rust-llama.cpp/pull/7)
- [x] Implement support for GPU (Metal)
- [ ] Add some test cases
- [ ] Support for fetching models through http & S3
- [x] Sync with latest master & support GGUF
- [x] Add some proper examples https://github.com/mdrokz/rust-llama.cpp/pull/7

## LICENSE

MIT
 
