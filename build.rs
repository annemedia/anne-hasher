extern crate cc;
extern crate winres;

fn main() {
    let mut base = cc::Build::new();
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let mut res = winres::WindowsResource::new();
        res.set_manifest(r#"
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="requireAdministrator" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
</assembly>
        "#);
        if let Err(e) = res.compile() {
            eprintln!("Error compiling Windows resource: {}", e);
            std::process::exit(1);
        }
    }

    if std::env::var("CARGO_CFG_TARGET_ENV").unwrap() == "msvc" {
        base.flag("/O2")
            .flag("/Oi")
            .flag("/Ot")
            .flag("/Oy")
            .flag("/GT")
            .flag("/GL");
    } else {
        base.flag("-std=c99");

        if std::env::var_os("NO_MTUNE_NATIVE").is_none() {
            base.flag("-mtune=native");
        }
    }

    let mut config = base.clone();
    config.file("src/c/sph_shabal.c")
          .file("src/c/common.c")
          .compile("shabal");

    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64" {

        let mut config = base.clone();
        if std::env::var("CARGO_CFG_TARGET_ENV").unwrap() != "msvc" {
            config.flag("-msse2");
        }
        config.file("src/c/mshabal_128_sse2.c")
              .file("src/c/noncegen_128_sse2.c")
              .compile("shabal_sse2");

        let mut config = base.clone();
        if std::env::var("CARGO_CFG_TARGET_ENV").unwrap() == "msvc" {
            config.flag("/arch:AVX2");
        } else {
            config.flag("-mavx2");
        }
        config.file("src/c/mshabal_256_avx2.c")
              .file("src/c/noncegen_256_avx2.c")
              .compile("shabal_avx2");

        let mut config = base.clone();
        if std::env::var("CARGO_CFG_TARGET_ENV").unwrap() == "msvc" {
            config.flag("/arch:AVX");
        } else {
            config.flag("-mavx");
        }
        config.file("src/c/mshabal_128_avx.c")
              .file("src/c/noncegen_128_avx.c")
              .compile("shabal_avx");

        let mut config = base.clone();
        if std::env::var("CARGO_CFG_TARGET_ENV").unwrap() == "msvc" {
            config.flag("/arch:AVX512");
        } else {
            config.flag("-mavx512f");
        }
        config.file("src/c/mshabal_512_avx512f.c")
              .file("src/c/noncegen_512_avx512f.c")
              .compile("shabal_avx512");
    }
}
