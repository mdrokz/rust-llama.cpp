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
    pub numa: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            context_size: 512,
            seed: 0,
            f16_memory: true,
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

pub struct PredictOptions {
    pub seed: i32,
    pub threads: i32,
    pub tokens: i32,
    pub top_k: i32,
    pub repeat: i32,
    pub batch: i32,
    pub n_keep: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub penalty: f32,
    pub f16_kv: bool,
    pub debug_mode: bool,
    pub stop_prompts: Vec<String>,
    pub ignore_eos: bool,

    pub tail_free_sampling_z: f32,
    pub typical_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_eta: f32,
    pub mirostat_tau: f32,
    pub penalize_nl: bool,
    pub logit_bias: String,
    pub token_callback: Option<Box<dyn Fn(String) -> bool + Send + 'static>>,
    // pub token_callback: Option<fn(String) -> bool>,
    pub path_prompt_cache: String,
    pub m_lock: bool,
    pub m_map: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,
    pub main_gpu: String,
    pub tensor_split: String,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            seed: -1,
            threads: 8,
            tokens: 128,
            top_k: 40,
            repeat: 64,
            batch: 512,
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
            token_callback: None,
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
    pub fn set_context(&mut self, context_size: i32) {
        self.context_size = context_size;
    }

    pub fn set_model_seed(&mut self, seed: i32) {
        self.seed = seed;
    }

    pub fn enable_f16_memory(&mut self) {
        self.f16_memory = true;
    }

    pub fn enable_embeddings(&mut self) {
        self.embeddings = true;
    }

    pub fn enable_m_lock(&mut self) {
        self.m_lock = true;
    }

    pub fn set_m_map(&mut self, m_map: bool) {
        self.m_map = m_map;
    }

    pub fn set_n_batch(&mut self, n_batch: i32) {
        self.n_batch = n_batch;
    }

    pub fn set_tensor_split(&mut self, tensor_split: String) {
        self.tensor_split = tensor_split;
    }

    pub fn set_gpu_layers(&mut self, n_gpu_layers: i32) {
        self.n_gpu_layers = n_gpu_layers;
    }

    pub fn set_main_gpu(&mut self, main_gpu: String) {
        self.main_gpu = main_gpu;
    }
}

impl PredictOptions {
    pub fn set_prediction_tensor_split(&mut self, tensor_split: String) {
        self.tensor_split = tensor_split;
    }

    pub fn set_prediction_main_gpu(&mut self, main_gpu: String) {
        self.main_gpu = main_gpu;
    }

    pub fn enable_f16_kv(&mut self) {
        self.f16_kv = true;
    }

    pub fn enable_debug_mode(&mut self) {
        self.debug_mode = true;
    }

    pub fn enable_prompt_cache_all(&mut self) {
        self.prompt_cache_all = true;
    }

    pub fn enable_prompt_cache_ro(&mut self) {
        self.prompt_cache_ro = true;
    }

    pub fn enable_m_lock(&mut self) {
        self.m_lock = true;
    }

    pub fn set_m_lock(&mut self, m_lock: bool) {
        self.m_lock = m_lock;
    }

    pub fn set_memory_map(&mut self, m_map: bool) {
        self.m_map = m_map;
    }

    pub fn set_token_callback(
        &mut self,
        token_callback: Option<Box<dyn Fn(String) -> bool + Send + 'static>>,
    ) {
        self.token_callback = token_callback;
    }
    // pub fn set_token_callback(&mut self, token_callback: Option<fn(String) -> bool>) {
    //     self.token_callback = token_callback;
    // }

    pub fn set_path_prompt_cache(&mut self, path_prompt_cache: String) {
        self.path_prompt_cache = path_prompt_cache;
    }

    pub fn set_seed(&mut self, seed: i32) {
        self.seed = seed;
    }

    pub fn set_threads(&mut self, threads: i32) {
        self.threads = threads;
    }

    pub fn set_tokens(&mut self, tokens: i32) {
        self.tokens = tokens;
    }

    pub fn set_top_k(&mut self, top_k: i32) {
        self.top_k = top_k;
    }

    pub fn set_repeat(&mut self, repeat: i32) {
        self.repeat = repeat;
    }

    pub fn set_batch(&mut self, batch: i32) {
        self.batch = batch;
    }

    pub fn set_n_keep(&mut self, n_keep: i32) {
        self.n_keep = n_keep;
    }

    pub fn set_top_p(&mut self, top_p: f32) {
        self.top_p = top_p;
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
    }

    pub fn set_penalty(&mut self, penalty: f32) {
        self.penalty = penalty;
    }

    pub fn set_tail_free_sampling_z(&mut self, tail_free_sampling_z: f32) {
        self.tail_free_sampling_z = tail_free_sampling_z;
    }

    pub fn set_typical_p(&mut self, typical_p: f32) {
        self.typical_p = typical_p;
    }

    pub fn set_frequency_penalty(&mut self, frequency_penalty: f32) {
        self.frequency_penalty = frequency_penalty;
    }

    pub fn set_presence_penalty(&mut self, presence_penalty: f32) {
        self.presence_penalty = presence_penalty;
    }

    pub fn set_mirostat(&mut self, mirostat: i32) {
        self.mirostat = mirostat;
    }

    pub fn set_mirostat_eta(&mut self, mirostat_eta: f32) {
        self.mirostat_eta = mirostat_eta;
    }

    pub fn set_mirostat_tau(&mut self, mirostat_tau: f32) {
        self.mirostat_tau = mirostat_tau;
    }

    pub fn enable_penalize_nl(&mut self) {
        self.penalize_nl = true;
    }

    pub fn set_logit_bias(&mut self, logit_bias: String) {
        self.logit_bias = logit_bias;
    }

    pub fn ignore_eos(&mut self) {
        self.ignore_eos = true;
    }
}
