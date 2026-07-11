use std::env;
use std::path::{Path, PathBuf};

/// Discover UCC include/lib dirs (same logic as ucc-rs build.rs).
/// Order: UCC_PREFIX → UCC_INCLUDE_DIR/UCC_LIB_DIR → $HOME/.local/ucc → common prefixes → /usr
fn discover_ucc() -> (PathBuf, PathBuf) {
    println!("cargo:rerun-if-env-changed=UCC_PREFIX");
    println!("cargo:rerun-if-env-changed=UCC_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=UCC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=HOME");

    if let Ok(prefix) = env::var("UCC_PREFIX") {
        let prefix = PathBuf::from(prefix);
        return (prefix.join("include"), prefix.join("lib"));
    }

    let include = env::var("UCC_INCLUDE_DIR").ok().map(PathBuf::from);
    let lib = env::var("UCC_LIB_DIR").ok().map(PathBuf::from);
    if let (Some(inc), Some(lib)) = (include, lib) {
        return (inc, lib);
    }

    // Check $HOME/.local/ucc (common for `./configure --prefix=$HOME/.local`)
    if let Ok(home) = env::var("HOME") {
        let home_prefix = PathBuf::from(&home).join(".local").join("ucc");
        let inc = home_prefix.join("include");
        let lib = home_prefix.join("lib");
        if inc.join("ucc").join("api").join("ucc.h").exists()
            || inc.join("ucc.h").exists()
            || lib.join("libucc.so").exists()
            || lib.join("libucc.so.1").exists()
        {
            return (inc, lib);
        }
    }

    let fixed_candidates = ["/usr", "/usr/local", "/opt/ucc"];
    for c in fixed_candidates {
        let p = Path::new(c);
        let inc = p.join("include");
        let lib = p.join("lib");
        if inc.join("ucc").join("api").join("ucc.h").exists()
            || inc.join("ucc.h").exists()
            || lib.join("libucc.so").exists()
            || lib.join("libucc.so.1").exists()
        {
            return (inc, lib);
        }
    }

    (PathBuf::from("/usr/include"), PathBuf::from("/usr/lib"))
}

fn main() {
    let (_include_dir, lib_dir) = discover_ucc();

    // Propagate UCC native lib search path and rpath to downstream binaries
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=ucc");
    // Use RPATH (not RUNPATH) so the library is found at runtime without LD_LIBRARY_PATH
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
