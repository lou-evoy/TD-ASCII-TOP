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

**Prerequisites**

- TouchDesigner 2025.30000+
- CUDA Toolkit 13.x (12.8+ for Blackwell / RTX 50)
- Visual Studio 2022 or 2026 (MSVC, *Desktop development with C++*)
- CMake ≥ 3.24

**Build (Release)**

From an *x64 Native Tools Command Prompt* (so `cl` and `nvcc` are on `PATH`):

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

Output: `build/AsciiTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`,
restart TouchDesigner, and add the node from **OP Create → Custom → "ASCII"**.
