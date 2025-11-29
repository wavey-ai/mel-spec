use std::env;

fn main() {
    // Only add CUDA linking hints when the `cuda` feature is enabled.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Compile CUDA helper kernels.
    cc::Build::new()
        .cuda(true)
        .file("src/cuda_kernels.cu")
        .flag("-std=c++14")
        .flag("-allow-unsupported-compiler")
        .flag("-ccbin=g++-13")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .compile("cuda_kernels");

    // Allow users to point at a non-default CUDA install.
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        println!("cargo:rustc-link-search=native={path}");
    } else if let Ok(path) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        println!("cargo:rustc-link-search=native={}/lib64", path);
    }

    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cudart");

    // Re-run if these change so the linker picks up new paths.
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
