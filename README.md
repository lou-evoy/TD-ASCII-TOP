# ASCII TOP — edge-preserving real-time ASCII for TouchDesigner (CUDA)

A custom TOP that converts an image to ASCII glyphs each frame while **preserving edges**:
flat regions map to a luminance-ramped glyph set, while contours found by a
Difference-of-Gaussians pass are drawn with direction-matched line glyphs (`|` `-` `/` `\`),
so the picture keeps its structure instead of dissolving into a flat character grid. This is a port of [Acerola's ASCII shader](https://github.com/GarrettGunnell/AcerolaFX).

## Demo

<!-- screenshots / GIFs / video go here -->
*Coming soon.*

## Why this one

- **Edge preservation.** A Difference-of-Gaussians + Sobel-direction pass lays line glyphs
  *along* the contours, so structure (faces, silhouettes, hard edges) survives instead of
  washing out the way brightness-only ASCII does.
- **Fused kernel.** The full pipeline runs in a single CUDA kernel — roughly one read and one
  write per pixel.
- **Adjustable** glyph set, cell density, and source-color tinting.

## Getting the node

The compiled plugin isn't distributed in this repo. Precompiled builds will be available to
supporters on **Patreon** *(link coming soon)*. If you'd rather compile it yourself, read on.

## Build it yourself

**Requirements:** TouchDesigner 2025.32050 (TOP API v12), CUDA Toolkit 12.8+, Visual Studio
2022/2026 (Desktop development with C++), CMake ≥ 3.24, an NVIDIA GPU (Turing / RTX 20 or newer).

The TouchDesigner C++ SDK headers (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`) are not in this
repo — they ship inside TouchDesigner at `<TD install>/Samples/CPlusPlus/CudaTOP`, and
`-DTD_SDK_DIR` must point there (the default below assumes a standard `C:/Program Files/Derivative`
install).

Run the build from the **x64 Native Tools Command Prompt for VS** (Start menu) — a plain
PowerShell or cmd window doesn't have `cl` / `nvcc` on `PATH`:

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

This produces `build/AsciiTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`
(or run `cmake --build build --target install_to_td` to do it in one step), restart
TouchDesigner, and add the node from **OP Create → Custom → "ASCII"**.

The build targets `sm_75`–`sm_120` and needs CUDA 12.8+ for `sm_120` (Blackwell / RTX 50). For
an older toolkit or GPU, override the architecture list — e.g.
`-DAX_CUDA_ARCHITECTURES="75-real;86-real;89-real"`; run `nvcc --list-gpu-code` to see what your
toolkit supports.
