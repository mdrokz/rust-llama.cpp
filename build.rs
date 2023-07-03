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
        .include("./llama.cpp/examples")
        // .file("./llama.cpp/llama.cpp")
        .include("./llama.cpp")
        // .file("./llama.cpp/examples/common.cpp")
        .cpp(true)
        .compile("binding");
}
