#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub context_size: i32,
    pub seed: i32,
    pub n_batch: i32,
    pub f16_memory: bool,
    pub m_lock: bool,
    pub m_map: bool,
    pub low_vram: bool,
    pub vocab_only: bool,
    pub embeddings: bool,
    pub n_gpu_layers: i32,
    pub main_gpu: String,
    pub tensor_split: String,
    pub numa: bool
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            context_size: 512,
            seed: 0,
            f16_memory: false,
            m_lock: false,
            embeddings: false,
            low_vram: false,
            vocab_only: false,
            m_map: true,
            n_batch: 0,
            numa: false,
            n_gpu_layers: 0,
            main_gpu: String::from(""),
            tensor_split: String::from(""),
        }
    }
}

struct PredictOptions {
    seed: i32,
    threads: i32,
    tokens: i32,
    top_k: i32,
    repeat: i32,
    batch: i32,
    n_keep: i32,
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

impl ModelOptions {
    fn set_context(&mut self, context_size: i32) {
        self.context_size = context_size;
    }

    fn set_model_seed(&mut self, seed: i32) {
        self.seed = seed;
    }

    fn enable_f16_memory(&mut self) {
        self.f16_memory = true;
    }

    fn enable_embeddings(&mut self) {
        self.embeddings = true;
    }

    fn enable_m_lock(&mut self) {
        self.m_lock = true;
    }

    fn set_m_map(&mut self, m_map: bool) {
        self.m_map = m_map;
    }

    fn set_n_batch(&mut self, n_batch: i32) {
        self.n_batch = n_batch;
    }

    fn set_tensor_split(&mut self, tensor_split: String) {
        self.tensor_split = tensor_split;
    }

    fn set_gpu_layers(&mut self, n_gpu_layers: i32) {
        self.n_gpu_layers = n_gpu_layers;
    }

    fn set_main_gpu(&mut self, main_gpu: String) {
        self.main_gpu = main_gpu;
    }
}

impl PredictOptions {
    fn set_prediction_tensor_split(&mut self, tensor_split: String) {
        self.tensor_split = tensor_split;
    }

    fn set_prediction_main_gpu(&mut self, main_gpu: String) {
        self.main_gpu = main_gpu;
    }

    fn enable_f16_kv(&mut self) {
        self.f16_kv = true;
    }

    fn enable_debug_mode(&mut self) {
        self.debug_mode = true;
    }

    fn enable_prompt_cache_all(&mut self) {
        self.prompt_cache_all = true;
    }

    fn enable_prompt_cache_ro(&mut self) {
        self.prompt_cache_ro = true;
    }

    fn enable_m_lock(&mut self) {
        self.m_lock = true;
    }

    fn set_m_lock(&mut self, m_lock: bool) {
        self.m_lock = m_lock;
    }

    fn set_memory_map(&mut self, m_map: bool) {
        self.m_map = m_map;
    }

    fn set_token_callback(&mut self, token_callback: Box<dyn Fn(String) -> bool>) {
        self.token_callback = token_callback;
    }

    fn set_path_prompt_cache(&mut self, path_prompt_cache: String) {
        self.path_prompt_cache = path_prompt_cache;
    }

    fn set_seed(&mut self, seed: i32) {
        self.seed = seed;
    }

    fn set_threads(&mut self, threads: i32) {
        self.threads = threads;
    }

    fn set_tokens(&mut self, tokens: i32) {
        self.tokens = tokens;
    }

    fn set_top_k(&mut self, top_k: i32) {
        self.top_k = top_k;
    }

    fn set_repeat(&mut self, repeat: i32) {
        self.repeat = repeat;
    }

    fn set_batch(&mut self, batch: i32) {
        self.batch = batch;
    }

    fn set_n_keep(&mut self, n_keep: i32) {
        self.n_keep = n_keep;
    }

    fn set_top_p(&mut self, top_p: f64) {
        self.top_p = top_p;
    }

    fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    fn set_penalty(&mut self, penalty: f64) {
        self.penalty = penalty;
    }

    fn set_tail_free_sampling_z(&mut self, tail_free_sampling_z: f64) {
        self.tail_free_sampling_z = tail_free_sampling_z;
    }

    fn set_typical_p(&mut self, typical_p: f64) {
        self.typical_p = typical_p;
    }

    fn set_frequency_penalty(&mut self, frequency_penalty: f64) {
        self.frequency_penalty = frequency_penalty;
    }

    fn set_presence_penalty(&mut self, presence_penalty: f64) {
        self.presence_penalty = presence_penalty;
    }

    fn set_mirostat(&mut self, mirostat: i32) {
        self.mirostat = mirostat;
    }

    fn set_mirostat_eta(&mut self, mirostat_eta: f64) {
        self.mirostat_eta = mirostat_eta;
    }

    fn set_mirostat_tau(&mut self, mirostat_tau: f64) {
        self.mirostat_tau = mirostat_tau;
    }

    fn enable_penalize_nl(&mut self) {
        self.penalize_nl = true;
    }

    fn set_logit_bias(&mut self, logit_bias: String) {
        self.logit_bias = logit_bias;
    }

    fn ignore_eos(&mut self) {
        self.ignore_eos = true;
    }
}
