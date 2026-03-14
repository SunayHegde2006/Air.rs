//! Build script — auto-detects Windows SDK and MSVC lib paths for the linker.
//!
//! On Windows, building outside a Visual Studio Developer Command Prompt
//! means the `LIB` env var is empty. This causes linker errors for system
//! libraries like `bcrypt.lib`, `kernel32.lib`, etc.
//!
//! This build script automatically finds the Windows SDK and MSVC toolchain
//! lib directories and adds them to the linker search path.

fn main() {
    #[cfg(target_os = "windows")]
    windows_sdk_link_search();
}

#[cfg(target_os = "windows")]
fn windows_sdk_link_search() {
    // Skip if LIB is already set (e.g., inside VS Developer Prompt)
    if std::env::var("LIB").map(|v| !v.is_empty()).unwrap_or(false) {
        return;
    }

    let arch = if cfg!(target_arch = "x86_64") { "x64" } else { "x86" };

    // ── Windows SDK (um + ucrt) ──────────────────────────────────────
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

    // ── MSVC toolchain lib ───────────────────────────────────────────
    for edition in &["Professional", "Enterprise", "Community", "BuildTools"] {
        let base = format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\MSVC",
            edition
        );
        if let Some(version) = find_latest_subdir(&base) {
            let lib = format!(r"{}\{}\lib\{}", base, version, arch);
            if std::path::Path::new(&lib).exists() {
                println!("cargo:rustc-link-search=native={}", lib);
                break;
            }
        }
    }
}

#[cfg(target_os = "windows")]
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

#[cfg(target_os = "windows")]
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
