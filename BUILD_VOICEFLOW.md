# VoiceFlow — Build Instructions (Windows)

## Prerequisites

1. **Rust** (MSVC toolchain): `rustup default stable-x86_64-pc-windows-msvc`
2. **Visual Studio 2022 Community** with:
   - MSVC v143 C++ build tools
   - Windows 11 SDK (10.0.22621+)
3. **Bun**: `scoop install bun`
4. **LLVM**: `scoop install llvm`
5. **CMake**: `scoop install cmake`

## Build

```cmd
set LIBCLANG_PATH=C:\Users\nneer\scoop\apps\llvm\current\bin
set PATH=C:\Users\nneer\scoop\apps\cmake\current\bin;%PATH%
bun install
cargo build --manifest-path src-tauri\Cargo.toml
```

## Dev Mode (frontend hot-reload + Rust backend)

```bash
bun run tauri dev --manifest-path src-tauri/Cargo.toml
```

## Models

On first run, the app will prompt you to download a model.
Recommended: **Parakeet V3** (~478MB) - supports 25 European languages including Polish.

Models are stored in: `%APPDATA%\com.pais.handy\models\`

## Architecture

- **Frontend**: React + TypeScript + Tailwind (Vite)
- **Backend**: Rust (Tauri v2)
- **Audio**: cpal + VAD (Voice Activity Detection)
- **Transcription**: transcribe-rs (Whisper.cpp + ONNX Runtime with DirectML)
- **Supported models**: Parakeet V2/V3, Whisper, Moonshine, SenseVoice, GigaAM, Canary
- **Auto-type**: Enigo (Win32 SendInput)
