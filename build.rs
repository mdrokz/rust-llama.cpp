use std::env;
use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("./binding.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");


    cc::Build::new()
    .include("./llama.cpp")
    .file("./llama.cpp/ggml.c")
    .shared_flag(false)
    .cpp(false)
    .compile("ggml");

    let out_dir = env::var("OUT_DIR").unwrap();
    let ggml_obj = PathBuf::from(out_dir).join("llama.cpp/ggml.o");

    cc::Build::new()
        .include("./llama.cpp/examples")
        .include("./llama.cpp")
        .object(ggml_obj)
        .file("./llama.cpp/llama.cpp")
        .file("./binding.cpp")
        .cpp(true)
        .compile("binding");
}
