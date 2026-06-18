/* ASCII TOP — CUDA algorithm interface (edge-preserving real-time ASCII mosaic).
 *
 * Plain C++17 contract between the TouchDesigner glue (AsciiTOP.cpp) and the CUDA
 * implementation (AsciiCUDA.cu). The glyph bitmaps come from the built-in pixel-font atlas
 * (GlyphAtlas.{h,cpp}) and are handed here as a coverage buffer to upload.
 *
 * Algorithm (per Acerola's ASCII shader, minus the depth/normal G-buffer passes a flat TOP
 * can't supply): luminance -> Difference of Gaussians -> Sobel edge direction -> per-cell
 * vote on a dominant edge direction -> if a cell is "edgy" draw the matching line glyph
 * (| - / \), else draw a fill glyph chosen by the cell's average luminance.
 *
 * EXECUTION is rebuilt for speed: instead of ~9 full-resolution passes with global-memory
 * round-trips, everything runs in ONE fused tile kernel. A thread block loads a pixel tile
 * (+ a small halo for the blur) into shared memory once, does luma/DoG/Sobel/vote/average
 * entirely in shared memory, then writes every output pixel of the tile by sampling the
 * glyph atlas. Global traffic is ~one read + one write of the image (bandwidth floor), so
 * cost barely grows with resolution — built for 4K/8K+.
 *
 * Cells are POWER-OF-TWO so cell/local indexing is bit-shift/mask (no division) and one
 * block maps cleanly to one cell up to 32x32.
 */
#ifndef ASCII_CUDA_H
#define ASCII_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace ascii {

// Selectable cell sizes (px). 8 = densest / smallest font, 128 = largest font.
// Power-of-two so `cell = px >> log2(size)` and `local = px & (size-1)`.
static constexpr int kCellSizes[]   = { 8, 16, 32, 64, 128 };
static constexpr int kNumCellSizes  = 5;

// Edge glyph slots, in this fixed order at the front of the atlas. The kernel maps a
// Sobel gradient angle to one of these.
enum EdgeGlyph {
    kEdgeVertical     = 0,  // |   (near-vertical edges)
    kEdgeHorizontal   = 1,  // -   (near-horizontal edges)
    kEdgeDiagonalFwd  = 2,  // /
    kEdgeDiagonalBack = 3,  // backslash
    kNumEdgeGlyphs    = 4
};

// In-block analysis is capped at this many samples per axis. For cells <= this the DoG runs
// per pixel; for 64/128 cells it subsamples (one direction vote per cell doesn't need
// per-pixel DoG over a 128px cell). Also bounds shared-memory usage.
static constexpr int kMaxAnalysisDim = 32;

// Caps the Gaussian blur radius so the shared-memory halo stays bounded.
static constexpr int kMaxBlurRadius  = 8;

struct Params
{
    int32_t  width  = 0;
    int32_t  height = 0;

    int32_t  cellSize = 16;          // power of two, one of kCellSizes (8 = densest)

    // Edge detection — Difference of Gaussians on luminance ----------------------------
    bool     drawEdges    = true;
    int32_t  kernelSize   = 2;       // blur truncation radius in samples (1..kMaxBlurRadius)
    float    blurSigma    = 1.25f;   // base gaussian sigma
    float    sigmaScale   = 1.25f;   // second gaussian sigma = blurSigma * sigmaScale
    float    dogThreshold = 0.005f;  // a pixel is an edge if the DoG response >= this
    float    edgeThreshold = 0.0f;   // fraction (0..1) of a cell's samples that must be
                                     // edges before the cell draws an edge glyph

    bool     viewEdges       = false; // debug: output the detected edge map instead of glyphs

    // Luminance fill -------------------------------------------------------------------
    bool     drawFill    = true;
    float    exposure    = 1.0f;     // multiply luminance before the ramp lookup
    float    attenuation = 1.0f;     // exponent on luminance
    bool     invert      = false;    // invert the luminance -> glyph relationship

    // Color ----------------------------------------------------------------------------
    float    fgColor[3]  = { 1.0f, 1.0f, 1.0f };  // ASCII color  (default white)
    float    bgColor[3]  = { 0.0f, 0.0f, 0.0f };  // background    (default black)
    bool     tintFromSource = false;              // color glyphs by the cell's avg color
    float    tintAmount  = 1.0f;                  // blend fg toward the source avg (0..1)

    bool     bgra        = true;     // BGRA8 vs RGBA8 channel order
    bool     bypass      = false;    // copy input straight through (native-bypass stand-in)
};

// JFIF luma weights, shared by host reference and device.
static inline __host__ __device__ float luma(float r, float g, float b)
{
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

class AsciiRenderer
{
public:
    AsciiRenderer() = default;
    ~AsciiRenderer();

    AsciiRenderer(const AsciiRenderer&) = delete;
    AsciiRenderer& operator=(const AsciiRenderer&) = delete;

    // Upload a freshly rasterized glyph atlas (host -> device texture). `coverage` is a
    // host buffer, row-major, width = (kNumEdgeGlyphs + numFill) * glyphPx, height =
    // glyphPx, single channel uint8 ink coverage (0..255). Glyph order along the strip:
    // [edges: V, H, /, \] then [fill: numFill glyphs sorted ascending by ink coverage].
    // Re-uploads (reallocating the array) only when the size/count changed.
    cudaError_t setGlyphAtlas(const uint8_t* coverage, int glyphPx, int numFill,
                              cudaStream_t stream, const char** outError);

    // Must be called between TOP_Context::beginCUDAOperations()/endCUDAOperations().
    cudaError_t process(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
                        const Params& p, cudaStream_t stream, const char** outError);

    int  glyphPx()  const { return myGlyphPx; }
    int  numFill()  const { return myNumFill; }
    bool hasAtlas() const { return myAtlasTex != 0; }

private:
    // Single coverage atlas (all glyphs in one horizontal strip) bound to a point-filtered
    // texture. Glyphs are rasterized at the exact cell size, so the render samples 1:1 —
    // no scaling artifacts. "Crisp vs smooth" is decided at host build time (threshold the
    // coverage), so the device side stays a plain point sample.
    cudaArray_t         myAtlasArr = nullptr;
    cudaTextureObject_t myAtlasTex = 0;
    int                 myGlyphPx  = 0;   // == cell size the atlas was rasterized at
    int                 myNumFill  = 0;   // number of fill-ramp glyphs in the atlas
};

} // namespace ascii

#endif // ASCII_CUDA_H
