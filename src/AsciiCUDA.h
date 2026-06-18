/* ASCII TOP — CUDA algorithm interface (edge-preserving ASCII mosaic).
 * C++17 contract with AsciiCUDA.cu; glyphs from built-in pixel-font atlas (GlyphAtlas.{h,cpp}).
 * per Acerola: luma -> DoG -> Sobel direction -> per-cell vote -> edge glyph (| - / \) or luma fill.
 * one fused tile kernel (tile+halo in shared mem); ~1 read + 1 write. cells pow2, 1 block/cell up to 32x32.
 */
#ifndef ASCII_CUDA_H
#define ASCII_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace ascii {

// cell sizes (px), pow2: 8 = densest, 128 = largest
static constexpr int kCellSizes[]   = { 8, 16, 32, 64, 128 };
static constexpr int kNumCellSizes  = 5;

// edge glyph slots (front of atlas); kernel maps Sobel angle to one
enum EdgeGlyph {
    kEdgeVertical     = 0,  // |   (near-vertical edges)
    kEdgeHorizontal   = 1,  // -   (near-horizontal edges)
    kEdgeDiagonalFwd  = 2,  // /
    kEdgeDiagonalBack = 3,  // backslash
    kNumEdgeGlyphs    = 4
};

// max analysis samples/axis; bigger cells subsample. also bounds shared mem
static constexpr int kMaxAnalysisDim = 32;

// caps blur radius (bounds shared-mem halo)
static constexpr int kMaxBlurRadius  = 8;

struct Params
{
    int32_t  width  = 0;
    int32_t  height = 0;

    int32_t  cellSize = 16;          // power of two, one of kCellSizes (8 = densest)

    // edge detection — DoG on luma
    bool     drawEdges    = true;
    int32_t  kernelSize   = 2;       // blur truncation radius, samples (1..kMaxBlurRadius)
    float    blurSigma    = 1.25f;   // base sigma
    float    sigmaScale   = 1.25f;   // second sigma = blurSigma * sigmaScale
    float    dogThreshold = 0.005f;  // edge if DoG response >= this
    float    edgeThreshold = 0.0f;   // fraction of cell samples that must be edges to draw an edge glyph

    bool     viewEdges       = false; // debug: output edge map

    // luma fill
    bool     drawFill    = true;
    float    exposure    = 1.0f;     // luma multiplier before ramp
    float    attenuation = 1.0f;     // exponent on luma
    bool     invert      = false;    // invert luma -> glyph

    // color
    float    fgColor[3]  = { 1.0f, 1.0f, 1.0f };  // ASCII color (white)
    float    bgColor[3]  = { 0.0f, 0.0f, 0.0f };  // background (black)
    bool     tintFromSource = false;              // color glyphs by cell avg
    float    tintAmount  = 1.0f;                  // blend fg toward source avg

    bool     bgra        = true;     // BGRA8 vs RGBA8
    bool     bypass      = false;    // bypass
};

// JFIF luma weights (host + device)
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

    // upload glyph atlas (host -> device texture). coverage: row-major single-channel uint8,
    // width (kNumEdgeGlyphs+numFill)*glyphPx. realloc only when size/count changes
    cudaError_t setGlyphAtlas(const uint8_t* coverage, int glyphPx, int numFill,
                              cudaStream_t stream, const char** outError);

    // call between begin/endCUDAOperations()
    cudaError_t process(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
                        const Params& p, cudaStream_t stream, const char** outError);

    int  glyphPx()  const { return myGlyphPx; }
    int  numFill()  const { return myNumFill; }
    bool hasAtlas() const { return myAtlasTex != 0; }

private:
    // one coverage strip bound to a point-filtered texture; glyphs at exact cell size (1:1 sample)
    cudaArray_t         myAtlasArr = nullptr;
    cudaTextureObject_t myAtlasTex = 0;
    int                 myGlyphPx  = 0;   // == cell size the atlas was rasterized at
    int                 myNumFill  = 0;   // fill-ramp glyph count
};

} // namespace ascii

#endif // ASCII_CUDA_H
