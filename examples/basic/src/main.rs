use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use std::io::Write;

fn main() {
    let model_options = ModelOptions::default();

    let llama = LLama::new(
        "../../models/tinyllama-chat/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into(),
        // "/home/ivan/github/ggerganov/llama/models/7B/ggml-model-q4_0.gguf".into(),
        &model_options,
    )
    .unwrap();

    let predict_options = PredictOptions {
        tokens: 400,
        threads: 8,
        top_k: 90,
        top_p: 0.86,
        token_callback: Some(Box::new(|token| {
            print!("{}", token);

            std::io::stdout().flush().unwrap();
            true
        })),
        ..Default::default()
    };

    let prompt = "Building a website can be done in 10 simple steps:\nStep 1:";
    println!();
    println!("{prompt}");
    llama.predict(prompt.into(), predict_options).unwrap();
}
