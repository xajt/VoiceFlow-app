@echo off
set "VULKAN_SDK=C:\Users\nneer\scoop\apps\vulkan\current"
set "LIBCLANG_PATH=C:\Users\nneer\scoop\apps\llvm\current\bin"
set "PATH=%VULKAN_SDK%\Bin;%LIBCLANG_PATH%;C:\Users\nneer\scoop\apps\cmake\current\bin;%PATH%"
cd /d "C:\Users\nneer\Desktop\VoiceFlow-temp"
cargo build --manifest-path src-tauri\Cargo.toml
