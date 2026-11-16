fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Point codegen at the vendored protoc rather than relying on one being
    // installed on the end user's machine (a bare `cargo build` previously
    // failed without a system protobuf-compiler).
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(&["proto/gepa.proto"], &["proto"])?;
    Ok(())
}
