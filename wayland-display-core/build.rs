use pkg_config;
use std::env;

fn main() {
    // Link GStreamer CUDA library
    if let Err(e) = pkg_config::Config::new()
        .atleast_version("1.24")
        .probe("gstreamer-cuda-1.0")
    {
        eprintln!(
            "Warning: gstreamer-cuda-1.0 not found via pkg-config: {}",
            e
        );
        eprintln!("Attempting to link manually...");

        // Fallback: try to link directly
        println!("cargo:rustc-link-lib=gstcuda-1.0");
    }

    // Link CUDA runtime
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    } else {
        // Common CUDA installation paths
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvrtc");

    // Link EGL
    println!("cargo:rustc-link-lib=EGL");

    // Rerun if build script changes
    println!("cargo:rerun-if-changed=build.rs");
}
