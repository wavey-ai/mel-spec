use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_home = find_cuda_home().expect("CUDA toolkit not found; set CUDA_HOME or CUDA_PATH");
    let nvcc = cuda_home.join("bin").join("nvcc");
    let nvcc_str = nvcc
        .to_str()
        .expect("CUDA toolkit path contains invalid UTF-8");

    let current_path = env::var_os("PATH").unwrap_or_default();
    let mut path_entries = vec![cuda_home.join("bin")];
    path_entries.extend(env::split_paths(&current_path));
    let merged_path = env::join_paths(path_entries).expect("failed to construct PATH");
    env::set_var("PATH", merged_path);
    env::set_var("CUDA_HOME", &cuda_home);

    let mut build = cc::Build::new();
    build.cuda(true);
    build.compiler(nvcc_str);
    build.no_default_flags(true);
    build.warnings(false);
    build.extra_warnings(false);
    build.file("src/cuda_kernels.cu");
    build.flag("-std=c++14");
    build.flag("-Xcompiler");
    build.flag("-fPIC");
    build.compile("cuda_kernels");

    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        println!("cargo:rustc-link-search=native={path}");
    } else {
        for path in candidate_cuda_lib_dirs(&cuda_home) {
            if path.exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
    }

    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cudart");
}

fn find_cuda_home() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(path) = env::var_os(var) {
            let path = PathBuf::from(path);
            if path.join("bin").join("nvcc").exists() {
                return Some(path);
            }
        }
    }

    for candidate in ["/usr/local/cuda", "/opt/cuda"] {
        let path = Path::new(candidate);
        if path.join("bin").join("nvcc").exists() {
            return Some(path.to_path_buf());
        }
    }

    None
}

fn candidate_cuda_lib_dirs(cuda_home: &Path) -> Vec<PathBuf> {
    vec![
        cuda_home.join("lib64"),
        cuda_home.join("targets").join("x86_64-linux").join("lib"),
    ]
}
