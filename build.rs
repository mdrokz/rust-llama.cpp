use std::env;
use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("./binding.h")
        .blocklist_function("tokenCallback")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    bindings
        .write_to_file(&out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let mut cx_flags = String::from("-Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -march=native -mtune=native");
    let mut cxx_flags = String::from("-Wall -Wdeprecated-declarations -Wunused-but-set-variable -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -march=native -mtune=native");

    // check if os is linux
    // if so, add -fPIC to cxx_flags
    if cfg!(target_os = "linux") {
        cx_flags.push_str(" -pthread");
        cxx_flags.push_str(" -fPIC -pthread");
    }

    let mut cbuild = &mut cc::Build::new();

    let mut ccbuild = &mut cc::Build::new();

    let mut ggml_type = String::new();

    ccbuild = ccbuild
        .include("./llama.cpp/examples")
        .include("./llama.cpp");

    if cfg!(feature = "opencl") {
        cbuild = cbuild.flag("-DGGML_USE_CLBLAST");
        ccbuild = ccbuild.flag("-DGGML_USE_CLBLAST");

        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=OpenCL");
            println!("cargo:rustc-link-lib=clblast");
        } else if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=framework=OpenCL");
            println!("cargo:rustc-link-lib=clblast");
        }

        ccbuild = ccbuild.file("./llama.cpp/ggml-opencl.cpp");

        ggml_type = "opencl".to_string();
    }

    for cx_flag in cx_flags.split(" ").into_iter() {
        cbuild = cbuild.flag(cx_flag);
    }

    cbuild
        .include("./llama.cpp")
        .file("./llama.cpp/ggml.c")
        .cpp(false)
        .compile("ggml");

    for cxx_flag in cxx_flags.split(" ").into_iter() {
        ccbuild = ccbuild.flag(cxx_flag);
    }

    let ggml_obj = PathBuf::from(&out_path).join("llama.cpp/ggml.o");

    ccbuild = ccbuild.object(ggml_obj);

    if !ggml_type.is_empty() {
        let ggml_feature_obj =
            PathBuf::from(&out_path).join(format!("llama.cpp/ggml-{}.o", ggml_type));
        ccbuild = ccbuild.object(ggml_feature_obj);
    }

    ccbuild
        .shared_flag(true)
        .file("./llama.cpp/examples/common.cpp")
        .file("./llama.cpp/llama.cpp")
        .file("./binding.cpp")
        .cpp(true)
        .compile("binding");
}
