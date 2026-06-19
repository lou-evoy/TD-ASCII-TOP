// CUDA ASCII-mosaic kernels
#include "AsciiCUDA.h"
#include "CudaCheck.h"

#include "device_launch_parameters.h"

namespace ascii {

static constexpr float AX_PI = 3.14159265358979323846f;

// shared-mem dims, worst case (A=32, R=8)
static constexpr int LUMA_DIM = kMaxAnalysisDim + 2 * kMaxBlurRadius;  // 48
static constexpr int EDGE_DIM = kMaxAnalysisDim;                       // 32

struct KParams
{
    int   W, H, C, A, step, R, glyphPx, glyphShift, numFill;
    int   drawEdges, drawFill, invert, bgra, tintFromSource;
    float sigma1, sigma2, dogThr, edgeThr, exposure, atten, tintAmount;
    float fg0, fg1, fg2, fg3, bg0, bg1, bg2, bg3;
    int   viewEdges;
};

static __host__ __device__ __forceinline__ int divUp(int a, int b) { return (a + b - 1) / b; }
static __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}
static __device__ __forceinline__ float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// pixel RGB as [0,1], border-clamped
static __device__ __forceinline__ void readRGBA(cudaSurfaceObject_t s, int x, int y, int bgra,
                                               float& r, float& g, float& b, float& a)
{
    uchar4 c;
    surf2Dread(&c, s, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
    if (bgra) { b = c.x * (1.0f / 255.0f); g = c.y * (1.0f / 255.0f); r = c.z * (1.0f / 255.0f); }
    else      { r = c.x * (1.0f / 255.0f); g = c.y * (1.0f / 255.0f); b = c.z * (1.0f / 255.0f); }
    a = c.w * (1.0f / 255.0f);
}

// Sobel gradient -> edge-glyph slot (Acerola angle buckets), or -1
static __device__ __forceinline__ int classifyDir(float gx, float gy)
{
    float mag = sqrtf(gx * gx + gy * gy);
    if (mag <= 1e-6f) return -1;
    float theta = atan2f(gy, gx);
    float a = fabsf(theta) / AX_PI;                  // 0..1
    if (a < 0.05f || a > 0.9f)            return kEdgeVertical;
    if (a > 0.45f && a < 0.55f)           return kEdgeHorizontal;
    if (a >= 0.05f && a <= 0.45f)         return (theta > 0.0f) ? kEdgeDiagonalFwd : kEdgeDiagonalBack;
    if (a >= 0.55f && a <= 0.9f)          return (theta > 0.0f) ? kEdgeDiagonalBack : kEdgeDiagonalFwd;
    return -1;
}

// full 32-lane warp sum (thread counts are multiples of 32)
static __device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// bypass: input -> output
__global__ void passthroughKernel(cudaSurfaceObject_t in, cudaSurfaceObject_t out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uchar4 c;
    surf2Dread(&c, in, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
    surf2Dwrite(c, out, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

__global__ void asciiKernel(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
                            cudaTextureObject_t atlas, KParams P)
{
    const int A    = P.A;
    const int R    = P.R;
    const int LD   = A + 2 * R;
    const int step = P.step;
    const int C    = P.C;

    const int cellX0 = blockIdx.x * C;
    const int cellY0 = blockIdx.y * C;

    const int tx = threadIdx.x, ty = threadIdx.y;     // 0..A-1
    const int tid = ty * A + tx;
    const int nthreads = A * A;

    __shared__ float  sLuma[LUMA_DIM * LUMA_DIM];
    __shared__ float2 sBlurH[LUMA_DIM * EDGE_DIM];   // h-blurred, both sigmas
    __shared__ float  sEdge[EDGE_DIM * EDGE_DIM];
    __shared__ float w1[2 * kMaxBlurRadius + 1];
    __shared__ float w2[2 * kMaxBlurRadius + 1];
    __shared__ int   buckets[kNumEdgeGlyphs];
    __shared__ int   edgeCnt;
    __shared__ float sumR, sumG, sumB, sumA, sumL;
    __shared__ int   sGlyph;
    __shared__ float sFgR, sFgG, sFgB, sFgA;

    // cell-local sample (neg = halo) -> image pixel
    auto sampleToPixel = [&](int s, int origin, int limit) -> int {
        int p = origin + s * step + step / 2;
        return clampi(p, 0, limit - 1);
    };

    if (tid == 0) { edgeCnt = 0; sumR = sumG = sumB = sumA = sumL = 0.0f; }
    if (tid < kNumEdgeGlyphs) buckets[tid] = 0;
    __syncthreads();

    // 1a. each thread loads its inner sample's color, luma into sLuma; color kept in
    //     registers for warp-reduced average (cheaper than per-thread atomics)
    float myR, myG, myB, myA, myL;
    {
        int px = sampleToPixel(tx, cellX0, P.W);
        int py = sampleToPixel(ty, cellY0, P.H);
        readRGBA(inSurf, px, py, P.bgra, myR, myG, myB, myA);
        myL = luma(myR, myG, myB);
        sLuma[(R + ty) * LD + (R + tx)] = myL;
    }

    // cell color/luma avg: warp reduce, one atomic per warp
    {
        float rr = warpReduceSum(myR), gg = warpReduceSum(myG);
        float bb = warpReduceSum(myB), aa = warpReduceSum(myA), ll = warpReduceSum(myL);
        if ((tid & 31) == 0)
        { atomicAdd(&sumR, rr); atomicAdd(&sumG, gg); atomicAdd(&sumB, bb); atomicAdd(&sumA, aa); atomicAdd(&sumL, ll); }
    }

    // 1b. load halo ring (luma only)
    for (int k = tid; k < LD * LD; k += nthreads)
    {
        int lx = k % LD, ly = k / LD;
        if (lx >= R && lx < R + A && ly >= R && ly < R + A) continue;   // inner done above
        int px = sampleToPixel(lx - R, cellX0, P.W);
        int py = sampleToPixel(ly - R, cellY0, P.H);
        float r, g, b, a;
        readRGBA(inSurf, px, py, P.bgra, r, g, b, a);
        sLuma[ly * LD + lx] = luma(r, g, b);
    }

    // 2. normalized 1D Gaussian weights, both sigmas
    if (tid == 0)
    {
        float s1 = 0.0f, s2 = 0.0f;
        float inv1 = 1.0f / (2.0f * P.sigma1 * P.sigma1);
        float inv2 = 1.0f / (2.0f * P.sigma2 * P.sigma2);
        for (int k = -R; k <= R; ++k)
        {
            float g1 = expf(-(k * k) * inv1), g2 = expf(-(k * k) * inv2);
            w1[k + R] = g1; w2[k + R] = g2; s1 += g1; s2 += g2;
        }
        for (int k = 0; k < 2 * R + 1; ++k) { w1[k] /= s1; w2[k] /= s2; }
    }
    __syncthreads();

    // 3a. separable Gaussian, horizontal pass; both sigmas packed in float2
    for (int k = tid; k < LD * A; k += nthreads)
    {
        int x = k % A, y = k / A;
        float a1 = 0.0f, a2 = 0.0f;
        for (int di = -R; di <= R; ++di)
        {
            float lv = sLuma[y * LD + (R + x + di)];
            a1 += w1[di + R] * lv; a2 += w2[di + R] * lv;
        }
        sBlurH[y * A + x] = make_float2(a1, a2);
    }
    __syncthreads();

    // 3b. vertical pass + DoG -> binary edge
    {
        float b1 = 0.0f, b2 = 0.0f;
        for (int dj = -R; dj <= R; ++dj)
        {
            float2 hv = sBlurH[(R + ty + dj) * A + tx];
            b1 += w1[dj + R] * hv.x; b2 += w2[dj + R] * hv.y;
        }
        float D = b1 - b2;   // DoG, tau = 1
        float edge = (D >= P.dogThr) ? 1.0f : 0.0f;
        sEdge[ty * A + tx] = edge;
        float ecnt = warpReduceSum(edge);
        if ((tid & 31) == 0) atomicAdd(&edgeCnt, (int)ecnt);
    }
    __syncthreads();

    // 4. Sobel on edge map -> direction vote (edge samples only)
    {
        if (sEdge[ty * A + tx] > 0.0f)
        {
            int xm = max(tx - 1, 0), xp = min(tx + 1, A - 1);
            int ym = max(ty - 1, 0), yp = min(ty + 1, A - 1);
            float tl = sEdge[ym * A + xm], tc = sEdge[ym * A + tx], tr = sEdge[ym * A + xp];
            float ml = sEdge[ty * A + xm],                         mr = sEdge[ty * A + xp];
            float bl = sEdge[yp * A + xm], bc = sEdge[yp * A + tx], br = sEdge[yp * A + xp];
            float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
            float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
            int dir = classifyDir(gx, gy);
            if (dir >= 0) atomicAdd(&buckets[dir], 1);
        }
    }
    __syncthreads();

    // 5. thread 0 picks the cell glyph
    if (tid == 0)
    {
        int total = A * A;
        float invCount = 1.0f / (float)total;

        int dominant = 0, best = buckets[0];
        for (int j = 1; j < kNumEdgeGlyphs; ++j)
            if (buckets[j] > best) { best = buckets[j]; dominant = j; }

        bool edgy = P.drawEdges && best > 0 &&
                    (edgeCnt >= (int)(P.edgeThr * total + 0.5f));

        int glyph = -1;
        if (edgy)
        {
            glyph = dominant;                                   // 0..3 edge glyph
        }
        else if (P.drawFill && P.numFill > 0)
        {
            float l = sumL * invCount;
            l = clampf(powf(l * P.exposure, P.atten), 0.0f, 1.0f);
            if (P.invert) l = 1.0f - l;
            int level = (int)(l * P.numFill);
            if (level >= P.numFill) level = P.numFill - 1;
            if (level < 0) level = 0;
            glyph = kNumEdgeGlyphs + level;
        }

        float fr = P.fg0, fg = P.fg1, fb = P.fg2, fa = P.fg3;
        if (P.tintFromSource)
        {
            float ar = sumR * invCount, ag = sumG * invCount, ab = sumB * invCount, aa = sumA * invCount;
            fr = fr + (ar - fr) * P.tintAmount;
            fg = fg + (ag - fg) * P.tintAmount;
            fb = fb + (ab - fb) * P.tintAmount;
            fa = fa + (aa - fa) * P.tintAmount;
        }
        sGlyph = glyph; sFgR = fr; sFgG = fg; sFgB = fb; sFgA = fa;
    }
    __syncthreads();

    // 6. render every output pixel; glyph nearest-upscaled by glyphShift=log2(C/glyphPx),
    //    exact pow2 blow-up so it stays pixel-perfect
    const int glyph = sGlyph;
    const float bg0 = P.bg0, bg1 = P.bg1, bg2 = P.bg2, bg3 = P.bg3;
    const float fr = sFgR, fg = sFgG, fb = sFgB, fa = sFgA;
    const int glyphCol = (glyph >= 0) ? glyph * P.glyphPx : 0;

    for (int ly = ty; ly < C; ly += A)
        for (int lx = tx; lx < C; lx += A)
        {
            int ox = cellX0 + lx, oy = cellY0 + ly;
            if (ox >= P.W || oy >= P.H) continue;

            // debug: edge map (white = edge sample)
            if (P.viewEdges)
            {
                int sx = min(lx / step, A - 1), sy = min(ly / step, A - 1);
                uint32_t v = (sEdge[sy * A + sx] > 0.0f) ? 255u : 0u;
                uint32_t e = v | (v << 8) | (v << 16) | (255u << 24);
                surf2Dwrite(e, outSurf, ox * (int)sizeof(uchar4), oy, cudaBoundaryModeZero);
                continue;
            }

            float cov = 0.0f;
            if (glyph >= 0)
            {
                int gx = lx >> P.glyphShift;
                int gy = ly >> P.glyphShift;
                // TD surface is bottom-up: flip fill glyphs (edge glyphs are self-consistent)
                if (glyph >= kNumEdgeGlyphs) gy = (P.glyphPx - 1) - gy;
                cov = tex2D<float>(atlas, glyphCol + gx + 0.5f, gy + 0.5f);
            }

            float rr = bg0 + (fr - bg0) * cov;
            float gg = bg1 + (fg - bg1) * cov;
            float bb = bg2 + (fb - bg2) * cov;
            float aa = bg3 + (fa - bg3) * cov;

            uint32_t r8 = (uint32_t)clampf(rr * 255.0f + 0.5f, 0.0f, 255.0f);
            uint32_t g8 = (uint32_t)clampf(gg * 255.0f + 0.5f, 0.0f, 255.0f);
            uint32_t b8 = (uint32_t)clampf(bb * 255.0f + 0.5f, 0.0f, 255.0f);
            uint32_t a8 = (uint32_t)clampf(aa * 255.0f + 0.5f, 0.0f, 255.0f);
            uint32_t out = P.bgra ? (b8 | (g8 << 8) | (r8 << 16) | (a8 << 24))
                                  : (r8 | (g8 << 8) | (b8 << 16) | (a8 << 24));
            surf2Dwrite(out, outSurf, ox * (int)sizeof(uchar4), oy, cudaBoundaryModeZero);
        }
}

AsciiRenderer::~AsciiRenderer()
{
    if (myAtlasTex) cudaDestroyTextureObject(myAtlasTex);
    if (myAtlasArr) cudaFreeArray(myAtlasArr);
}

cudaError_t AsciiRenderer::setGlyphAtlas(const uint8_t* coverage, int glyphPx, int numFill,
                                         cudaStream_t stream, const char** outError)
{
    if (outError) *outError = nullptr;
    const int total  = kNumEdgeGlyphs + numFill;
    const int stripW = total * glyphPx;
    const int stripH = glyphPx;

    // realloc only when dims changed
    if (myAtlasArr == nullptr || myGlyphPx != glyphPx || myNumFill != numFill)
    {
        if (myAtlasTex) { cudaDestroyTextureObject(myAtlasTex); myAtlasTex = 0; }
        if (myAtlasArr) { cudaFreeArray(myAtlasArr); myAtlasArr = nullptr; }

        cudaChannelFormatDesc cd = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        AX_CUDA_RETURN(cudaMallocArray(&myAtlasArr, &cd, stripW, stripH), outError);

        cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = myAtlasArr;

        cudaTextureDesc td; memset(&td, 0, sizeof(td));
        td.addressMode[0] = cudaAddressModeClamp;
        td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode     = cudaFilterModePoint;
        td.readMode       = cudaReadModeNormalizedFloat;   // uint8 -> [0,1]
        td.normalizedCoords = 0;
        AX_CUDA_RETURN(cudaCreateTextureObject(&myAtlasTex, &rd, &td, nullptr), outError);

        myGlyphPx = glyphPx; myNumFill = numFill;
    }

    AX_CUDA_RETURN(cudaMemcpy2DToArrayAsync(myAtlasArr, 0, 0, coverage,
                       stripW, stripW, stripH, cudaMemcpyHostToDevice, stream), outError);
    return cudaSuccess;
}

cudaError_t AsciiRenderer::process(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
                                   const Params& p, cudaStream_t stream, const char** outError)
{
    if (outError) *outError = nullptr;
    const int W = p.width, H = p.height;
    if (!inSurf) return cudaSuccess;

    if (p.bypass)
    {
        dim3 b(16, 16, 1), g(divUp(W, 16), divUp(H, 16), 1);
        passthroughKernel<<<g, b, 0, stream>>>(inSurf, outSurf, W, H);
        AX_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    if (!hasAtlas())
    {
        // no atlas yet: pass through instead of black
        dim3 b(16, 16, 1), g(divUp(W, 16), divUp(H, 16), 1);
        passthroughKernel<<<g, b, 0, stream>>>(inSurf, outSurf, W, H);
        AX_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    int C = p.cellSize;
    if (C < kCellSizes[0]) C = kCellSizes[0];
    const int A    = (C < kMaxAnalysisDim) ? C : kMaxAnalysisDim;
    const int step = C / A;

    float sigma1 = p.blurSigma     < 0.1f ? 0.1f : p.blurSigma;
    float sigma2 = sigma1 * (p.sigmaScale < 1.0f ? 1.0f : p.sigmaScale);
    int   R = p.kernelSize;                 // truncation radius, samples
    if (R < 1) R = 1;
    if (R > kMaxBlurRadius) R = kMaxBlurRadius;

    // glyphShift = log2(C / glyphPx); atlas built at glyphPx <= C
    int gp = myGlyphPx < 1 ? 1 : myGlyphPx;
    int glyphShift = 0;
    for (int t = C / gp; t > 1; t >>= 1) ++glyphShift;

    KParams kp;
    kp.W = W; kp.H = H; kp.C = C; kp.A = A; kp.step = step; kp.R = R;
    kp.glyphPx = myGlyphPx; kp.glyphShift = glyphShift; kp.numFill = myNumFill;
    kp.drawEdges = p.drawEdges ? 1 : 0; kp.drawFill = p.drawFill ? 1 : 0;
    kp.invert = p.invert ? 1 : 0; kp.bgra = p.bgra ? 1 : 0;
    kp.tintFromSource = p.tintFromSource ? 1 : 0;
    kp.sigma1 = sigma1; kp.sigma2 = sigma2; kp.dogThr = p.dogThreshold;
    kp.edgeThr = p.edgeThreshold; kp.exposure = p.exposure; kp.atten = p.attenuation;
    kp.tintAmount = p.tintAmount;
    kp.fg0 = p.fgColor[0]; kp.fg1 = p.fgColor[1]; kp.fg2 = p.fgColor[2]; kp.fg3 = p.fgColor[3];
    kp.bg0 = p.bgColor[0]; kp.bg1 = p.bgColor[1]; kp.bg2 = p.bgColor[2]; kp.bg3 = p.bgColor[3];
    kp.viewEdges = p.viewEdges ? 1 : 0;

    dim3 block(A, A, 1);
    dim3 grid(divUp(W, C), divUp(H, C), 1);
    asciiKernel<<<grid, block, 0, stream>>>(inSurf, outSurf, myAtlasTex, kp);
    AX_CUDA_CHECK_LAUNCH(outError);
    return cudaSuccess;
}

} // namespace ascii
