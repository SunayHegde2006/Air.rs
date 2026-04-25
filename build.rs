//! Build script -- Cross-platform linker configuration for Air.rs
//!
//! Handles three platforms:
//!   - Windows (MSVC): auto-detects Windows SDK + MSVC toolchain lib paths,
//!     creates a stub stdc++.lib for candle-flash-attn compatibility.
//!   - macOS: links system frameworks (Metal, Accelerate) when features enabled.
//!   - Linux: links stdc++ for flash-attn CUDA kernels.

fn main() {
    // --- Windows MSVC -------------------------------------------------------
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        windows_sdk_link_search();
        create_stdc_stub();
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

        // CUDA paths (if $CUDA_HOME or /usr/local/cuda exist)
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
            } else if std::path::Path::new("/usr/local/cuda/lib64").exists() {
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            }
        }
    }

    // Tell Cargo to re-run if these change
    println!("cargo:rerun-if-env-changed=LIB");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}

// ===========================================================================
// Windows helpers
// ===========================================================================

/// Create a valid empty `stdc++.lib` so MSVC linker doesn't fail when
/// candle-flash-attn emits `cargo:rustc-link-lib=stdc++`.
///
/// Strategy 1: Use MSVC `lib.exe` to create a proper empty .lib
/// Strategy 2: Write a minimal COFF object file and archive it manually
#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn create_stdc_stub() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let stub_path = std::path::Path::new(&out_dir).join("stdc++.lib");

    // Always recreate (in case an old invalid stub exists)
    if stub_path.exists() {
        let _ = std::fs::remove_file(&stub_path);
    }

    // Strategy 1: Use lib.exe (always available when MSVC is installed)
    let lib_result = std::process::Command::new("lib.exe")
        .args(["/NOLOGO", "/MACHINE:X64", &format!("/OUT:{}", stub_path.display())])
        .output();

    if let Ok(output) = lib_result {
        if output.status.success() && stub_path.exists() && stub_path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
            println!("cargo:rustc-link-search=native={}", out_dir);
            return;
        }
    }

    // Strategy 2: Create a minimal COFF .obj and archive it with lib.exe
    let obj_path = std::path::Path::new(&out_dir).join("empty.obj");
    // Minimal COFF object: header (20 bytes) with 0 sections, 0 symbols
    let mut coff: Vec<u8> = Vec::with_capacity(20);
    coff.extend_from_slice(&0x8664u16.to_le_bytes()); // Machine: AMD64
    coff.extend_from_slice(&0u16.to_le_bytes());       // NumberOfSections: 0
    coff.extend_from_slice(&0u32.to_le_bytes());       // TimeDateStamp
    coff.extend_from_slice(&0u32.to_le_bytes());       // PointerToSymbolTable
    coff.extend_from_slice(&0u32.to_le_bytes());       // NumberOfSymbols
    coff.extend_from_slice(&0u16.to_le_bytes());       // SizeOfOptionalHeader
    coff.extend_from_slice(&0u16.to_le_bytes());       // Characteristics
    std::fs::write(&obj_path, &coff).unwrap();

    let lib_result = std::process::Command::new("lib.exe")
        .args([
            "/NOLOGO",
            "/MACHINE:X64",
            &format!("/OUT:{}", stub_path.display()),
            &format!("{}", obj_path.display()),
        ])
        .output();

    if let Ok(output) = lib_result {
        if output.status.success() && stub_path.exists() {
            let _ = std::fs::remove_file(&obj_path);
            println!("cargo:rustc-link-search=native={}", out_dir);
            return;
        }
    }

    // Strategy 3: Last resort — no lib.exe available at all.
    // This should never happen on a properly configured MSVC system,
    // but if it does, the linker will fail with a clear error about stdc++.lib.
    eprintln!("cargo:warning=Could not create stdc++.lib stub. Build may fail if flash-attn is enabled.");
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
