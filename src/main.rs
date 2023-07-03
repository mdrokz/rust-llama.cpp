use std::{
    error::Error,
    ffi::{c_void, CString},
};

use options::ModelOptions;

mod options;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[derive(Debug, Clone, Copy)]
struct LLama {
    state: *mut c_void,
    embeddings: bool,
    contextSize: i32,
}

impl LLama {
    pub fn new(model: String, opts: ModelOptions) -> Result<Self, Box<dyn Error>> {
        let model_path = CString::new(model).unwrap();
        unsafe {
            let result = load_model(
                model_path.as_ptr(),
                opts.context_size,
                opts.seed,
                opts.f16_memory,
                opts.m_lock,
                opts.embeddings,
                opts.m_map,
                opts.low_vram,
                opts.vocab_only,
                opts.n_gpu_layers,
                opts.n_batch,
                CString::new(opts.main_gpu).unwrap().as_ptr(),
                CString::new(opts.tensor_split).unwrap().as_ptr(),
                opts.numa,
            );

            println!("result: {:?}", result == std::ptr::null_mut());
            if result == std::ptr::null_mut() {
                return Err("Failed to load model".into());
            } else {
                Ok(Self {
                    state: result,
                    embeddings: opts.embeddings,
                    contextSize: opts.context_size,
                })
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts = ModelOptions::default();

    let llama = LLama::new(String::from("/home/mdrokz/Documents/Projects/docker/vicuna/models2/wizard-vicuna-13B.ggmlv3.q4_0.bin
    "), opts)?;
    // let llama = LLama::new(String::from("./wizard-vicuna-13B.ggmlv3.q4_0.bin"), opts);

    println!("Hello, world!");

    Ok(())
}
