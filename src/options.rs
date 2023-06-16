#[derive(Debug, Clone)]
struct ModelOptions {
    context_size: i64,
    seed: i64,
    n_batch: i64,
    f16_memory: bool,
    m_lock: bool,
    m_map: bool,
    embeddings: bool,
    n_gpu_layers: i64,
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

struct PredictOptions {
    seed: i64,
    threads: i64,
    tokens: i64,
    top_k: i64,
    repeat: i64,
    batch: i64,
    n_keep: i64,
    top_p: f64,
    temperature: f64,
    penalty: f64,
    f16_kv: bool,
    debug_mode: bool,
    stop_prompts: Vec<String>,
    ignore_eos: bool,

    tail_free_sampling_z: f64,
    typical_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    mirostat: i32,
    mirostat_eta: f64,
    mirostat_tau: f64,
    penalize_nl: bool,
    logit_bias: String,
    token_callback: Box<dyn Fn(String) -> bool>,

    path_prompt_cache: String,
    m_lock: bool,
    m_map: bool,
    prompt_cache_all: bool,
    prompt_cache_ro: bool,
    main_gpu: String,
    tensor_split: String,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            seed: -1,
            threads: 4,
            tokens: 128,
            top_k: 40,
            repeat: 64,
            batch: 8,
            n_keep: 64,
            top_p: 0.95,
            temperature: 0.8,
            penalty: 1.1,
            f16_kv: false,
            debug_mode: false,
            stop_prompts: vec![],
            ignore_eos: false,
            tail_free_sampling_z: 1.0,
            typical_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0,
            mirostat_eta: 0.1,
            mirostat_tau: 5.0,
            penalize_nl: false,
            logit_bias: String::from(""),
            token_callback: Box::new(|_| true),
            path_prompt_cache: String::from(""),
            m_lock: false,
            m_map: false,
            prompt_cache_all: false,
            prompt_cache_ro: false,
            main_gpu: String::from(""),
            tensor_split: String::from(""),
        }
    }
}