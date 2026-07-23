//! Build script -- Cross-platform linker configuration for Air.rs
//!
//! Handles three platforms:
//!   - Windows (MSVC): auto-detects Windows SDK + MSVC toolchain lib paths,
//!     creates a stub stdc++.lib for candle-flash-attn compatibility.
//!   - macOS: links system frameworks (Metal, Accelerate) when features enabled.
//!   - Linux: links stdc++ for flash-attn CUDA kernels.
//!
//! Also compiles:
//!   - src/kernels/air_compute_impl.c  → libair_compute (AVX-512/AVX2/scalar)
//!   - src/kernels/*.comp              → *.spv (Vulkan compute, requires glslc)

fn main() {
    // ── CPU compute kernels (AVX-512 / AVX2 / scalar fallback) ─────────────
    // Compiled unconditionally — used by all three hardware backends as the
    // host-side activation pre/post-processing layer.
    compile_cpu_kernels();

    // ── Vulkan compute shaders → SPIR-V (requires glslc from Vulkan SDK) ──
    #[cfg(feature = "vulkan")]
    compile_vulkan_shaders();

    // --- Windows MSVC -------------------------------------------------------
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        windows_sdk_link_search();
        create_msvc_stubs(&["stdc++", "amdhip64", "cufile"]);
    }

    // --- macOS --------------------------------------------------------------
    #[cfg(target_os = "macos")]
    {
        // Metal framework for GPU compute (ucal.rs MetalContext)
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        }

        // Accelerate framework for BLAS/LAPACK (faster matmul on Apple Silicon)
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // --- Linux --------------------------------------------------------------
    #[cfg(target_os = "linux")]
    {
        // flash-attn needs libstdc++
        #[cfg(feature = "flash-attn")]
        println!("cargo:rustc-link-lib=stdc++");

        // CUDA paths and architecture targeting
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
            } else if std::path::Path::new("/usr/local/cuda/lib64").exists() {
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            }

            // Detect GPU compute capability via nvidia-smi and export NVCC_ARCH
            // so the scripts/nvcc wrapper compiles kernels for the actual GPU ISA
            // (e.g. sm_89 for Ada Lovelace, sm_90 for Hopper, sm_100 for Blackwell).
            // Falls back gracefully if nvidia-smi is unavailable (Docker / CI).
            detect_and_export_cuda_arch();
        }

        // Vulkan Linking (STRIX Protocol §12.1)
        // If libvulkan.so is missing (common on minimal CI), fallback to libvulkan.so.1
        #[cfg(feature = "vulkan")]
        {
            let has_dev_link = std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists() ||
                              std::path::Path::new("/usr/lib64/libvulkan.so").exists();
            if has_dev_link {
                println!("cargo:rustc-link-lib=vulkan");
            } else {
                // Link against the versioned library directly
                println!("cargo:rustc-link-arg=-l:libvulkan.so.1");
            }
        }
    }

    // Tell Cargo to re-run if these change
    println!("cargo:rerun-if-env-changed=LIB");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC_ARCH");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");
    println!("cargo:rerun-if-changed=src/kernels/air_compute_impl.c");
    println!("cargo:rerun-if-changed=src/kernels/air_rmsnorm.comp");
    println!("cargo:rerun-if-changed=src/kernels/air_matmul.comp");
}

/// Compile src/kernels/air_compute_impl.c into a static library linked into
/// the Rust binary. Enables AVX-512/AVX2 if the host compiler supports it.
fn compile_cpu_kernels() {
    let mut build = cc::Build::new();
    build
        .file("src/kernels/air_compute_impl.c")
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffast-math")
        // Link math library for sqrtf / cosf / sinf / powf
        .flag_if_supported("-lm");
    build.compile("air_compute");
    // Emit the link instruction so rustc can find the archive
    println!("cargo:rustc-link-lib=static=air_compute");
}

/// Compile Vulkan GLSL compute shaders to SPIR-V using glslc (Vulkan SDK).
/// If glslc is not in PATH this step is skipped with a warning — the
/// VulkanBlock will fall back to the CPU path at runtime.
#[allow(dead_code)]
fn compile_vulkan_shaders() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shaders = &[
        "src/kernels/air_rmsnorm.comp",
        "src/kernels/air_matmul.comp",
    ];

    // Locate glslc
    let glslc = if let Ok(g) = std::env::var("GLSLC") {
        std::path::PathBuf::from(g)
    } else {
        // Search standard Vulkan SDK locations
        let candidates = [
            "/usr/bin/glslc",
            "/usr/local/bin/glslc",
            "/usr/lib/x86_64-linux-gnu/glslc",
        ];
        match candidates.iter().find(|p| std::path::Path::new(p).exists()) {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                println!("cargo:warning=Air.rs: glslc not found — Vulkan shaders will not be compiled. Install the Vulkan SDK or set GLSLC env var.");
                return;
            }
        }
    };

    for src in shaders {
        let stem = std::path::Path::new(src)
            .file_stem().unwrap()
            .to_string_lossy();
        let spv_path = format!("{}/{}.spv", out_dir, stem);
        let status = std::process::Command::new(&glslc)
            .args([
                "--target-env=vulkan1.2",
                "-O",
                src,
                "-o",
                &spv_path,
            ])
            .status()
            .expect("failed to spawn glslc");
        if !status.success() {
            panic!("glslc failed to compile {}", src);
        }
        println!("cargo:warning=Air.rs: compiled {} → {}", src, spv_path);
    }

    // Export the OUT_DIR so VulkanBlock can find the .spv files at runtime
    println!("cargo:rustc-env=AIR_SHADER_DIR={}", out_dir);
}

/// Detect the primary GPU's compute capability (e.g. "89" for sm_89) via
/// `nvidia-smi --query-gpu=compute_cap` and export it as `NVCC_ARCH` for the
/// `scripts/nvcc` wrapper to pick up, so all CUDA kernels (including those in
/// transitive dependencies like candle-flash-attn) are compiled for the actual
/// installed GPU ISA rather than NVCC's ancient sm_52 default.
///
/// Also emits a `CUDA_ARCH` compile-time env var readable in Rust code via
/// `env!("CUDA_ARCH")`.
#[allow(dead_code)]
#[cfg(feature = "cuda")]
fn detect_and_export_cuda_arch() {
    // Honour an explicit override first (useful in CI with a known GPU target)
    let arch_arg = if let Ok(arch) = std::env::var("NVCC_ARCH") {
        if !arch.is_empty() {
            Some(arch)
        } else {
            None
        }
    } else {
        None
    };

    let arch = match arch_arg {
        Some(explicit) => explicit,
        None => {
            // Query nvidia-smi for all GPUs
            let mut detected_arches = Vec::new();
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
            {
                if output.status.success() {
                    let raw = String::from_utf8_lossy(&output.stdout);
                    for line in raw.lines() {
                        let sm = line.trim().replace('.', "");
                        if !sm.is_empty() && sm.chars().all(|c| c.is_ascii_digit()) {
                            detected_arches.push(format!("sm_{}", sm));
                        }
                    }
                }
            }
            if !detected_arches.is_empty() {
                detected_arches.sort();
                detected_arches.dedup();
                detected_arches.join(",")
            } else {
                "".to_string()
            }
        }
    };

    if !arch.is_empty() {
        println!("cargo:rustc-env=CUDA_ARCH={}", arch);
        println!("cargo:warning=Air.rs: target GPU arch(es) detected/set: {arch}, injecting corresponding compiler flags via nvcc wrapper");
        // Write to OUT_DIR so scripts/nvcc can source it if needed
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let _ = std::fs::write(
                format!("{}/cuda_arch.txt", out_dir),
                &arch,
                );
        }
    } else {
        // Fallback: no GPU detected (CI, Docker without GPU passthrough)
        println!("cargo:warning=Air.rs: nvidia-smi not available and NVCC_ARCH not set; CUDA builds will default");
        println!("cargo:rustc-env=CUDA_ARCH=");
    }
}

#[allow(dead_code)]
#[cfg(not(feature = "cuda"))]
fn detect_and_export_cuda_arch() {}

// ===========================================================================
// Windows helpers
// ===========================================================================

/// Create a valid empty `stdc++.lib` so MSVC linker doesn't fail when
/// candle-flash-attn emits `cargo:rustc-link-lib=stdc++`.
///
/// Strategy 1: Use MSVC `lib.exe` to create a proper empty .lib
/// Strategy 2: Write a minimal COFF object file and archive it manually
/// Create valid empty `.lib` files for MSVC so the linker doesn't fail on CI
/// when GPU SDKs or Linux-originating libraries (stdc++) are missing.
#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn create_msvc_stubs(libs: &[&str]) {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj_path = std::path::Path::new(&out_dir).join("empty.obj");

    // 1. Create a minimal COFF .obj file
    // Minimal COFF header: Machine=AMD64, Sections=0, Symbols=0
    let mut coff: Vec<u8> = Vec::with_capacity(20);
    coff.extend_from_slice(&0x8664u16.to_le_bytes()); // Machine: AMD64
    coff.extend_from_slice(&0u16.to_le_bytes());       // NumberOfSections: 0
    coff.extend_from_slice(&0u32.to_le_bytes());       // TimeDateStamp
    coff.extend_from_slice(&0u32.to_le_bytes());       // PointerToSymbolTable
    coff.extend_from_slice(&0u32.to_le_bytes());       // NumberOfSymbols
    coff.extend_from_slice(&0u16.to_le_bytes());       // SizeOfOptionalHeader
    coff.extend_from_slice(&0u16.to_le_bytes());       // Characteristics
    std::fs::write(&obj_path, &coff).unwrap();

    // 2. Archive it into each requested .lib
    for lib_base in libs {
        let stub_path = std::path::Path::new(&out_dir).join(format!("{}.lib", lib_base));
        
        // Use lib.exe (always available when MSVC is installed)
        let _ = std::process::Command::new("lib.exe")
            .args([
                "/NOLOGO",
                "/MACHINE:X64",
                &format!("/OUT:{}", stub_path.display()),
                &format!("{}", obj_path.display()),
            ])
            .output();
    }

    let _ = std::fs::remove_file(&obj_path);
    println!("cargo:rustc-link-search=native={}", out_dir);
}

/// Auto-detect Windows SDK and MSVC toolchain library paths so builds
/// work outside a Visual Studio Developer Command Prompt.
#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn windows_sdk_link_search() {
    // Skip if LIB is already set (e.g., inside VS Developer Prompt)
    if std::env::var("LIB").map(|v| !v.is_empty()).unwrap_or(false) {
        return;
    }

    let arch = if cfg!(target_arch = "x86_64") { "x64" } else { "x86" };

    // -- Windows SDK (um + ucrt) --
    let sdk_root = r"C:\Program Files (x86)\Windows Kits\10\Lib";
    if let Some(version) = find_latest_sdk_version(sdk_root) {
        let um = format!(r"{}\{}\um\{}", sdk_root, version, arch);
        let ucrt = format!(r"{}\{}\ucrt\{}", sdk_root, version, arch);
        if std::path::Path::new(&um).exists() {
            println!("cargo:rustc-link-search=native={}", um);
        }
        if std::path::Path::new(&ucrt).exists() {
            println!("cargo:rustc-link-search=native={}", ucrt);
        }
    }

    // -- MSVC toolchain lib --
    for year in &["2022", "2019"] {
        for edition in &["Professional", "Enterprise", "Community", "BuildTools"] {
            let base = format!(
                r"C:\Program Files\Microsoft Visual Studio\{}\{}\VC\Tools\MSVC",
                year, edition
            );
            if let Some(version) = find_latest_subdir(&base) {
                let lib = format!(r"{}\{}\lib\{}", base, version, arch);
                if std::path::Path::new(&lib).exists() {
                    println!("cargo:rustc-link-search=native={}", lib);
                    return; // found it, done
                }
            }
        }
    }
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn find_latest_sdk_version(sdk_root: &str) -> Option<String> {
    let mut versions: Vec<String> = std::fs::read_dir(sdk_root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("10.0.") { Some(name) } else { None }
        })
        .collect();
    versions.sort();
    versions.pop()
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn find_latest_subdir(base: &str) -> Option<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    dirs.sort();
    dirs.pop()
}
