#[derive(Debug, Clone)]
struct ModelOptions {
    context_size: i32,
    seed: i32,
    n_batch: i32,
    f16_memory: bool,
    m_lock: bool,
    m_map: bool,
    embeddings: bool,
    n_gpu_layers: i32,
    main_gpu: String,
    tensor_split: String,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            context_size: 512,
            seed: 0,
            f16_memory: false,
            m_lock: false,
            embeddings: false,
            m_map: true,
            n_batch: 0,
            n_gpu_layers: 0,
            main_gpu: String::from(""),
            tensor_split: String::from(""),
        }
    }
}