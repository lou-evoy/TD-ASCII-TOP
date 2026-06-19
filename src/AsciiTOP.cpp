/* ASCII TOP — TD glue; mirrors the CudaTOP sample. */
#include "AsciiTOP.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <utility>

// plugin version
static const char* kVersion = "1.0.0";

extern "C"
{

DLLEXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CUDA;

    info->customOPInfo.opType->setString("Ascii");
    info->customOPInfo.opLabel->setString("ASCII");
    info->customOPInfo.opIcon->setString("ASC");
    info->customOPInfo.authorName->setString("SAT");
    info->customOPInfo.authorEmail->setString("levoy@sat.qc.ca");

    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;

    info->customOPInfo.majorVersion = 1;
    info->customOPInfo.minorVersion = 0;
}

DLLEXPORT TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new AsciiTOP(info, context);
}

DLLEXPORT void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context* context)
{
    delete (AsciiTOP*)instance;
}

} // extern "C"

// recreate each cook, never cache: bypass/reactivate frees the cudaArray (stale handle)
static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
    if (*surface)
    {
        cudaDestroySurfaceObject(*surface);
        *surface = 0;
    }
    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaCreateSurfaceObject(surface, &desc);
}

static bool
isSupported8BitRGBA(OP_PixelFormat f)
{
    return f == OP_PixelFormat::BGRA8Fixed || f == OP_PixelFormat::RGBA8Fixed;
}

AsciiTOP::AsciiTOP(const OP_NodeInfo* info, TOP_Context* context) :
    myNodeInfo(info), myContext(context), myStream(0),
    myInputSurface(0), myOutputSurface(0), myAtlasValid(false),
    myHasPending(false), myError(nullptr)
{
    cudaStreamCreate(&myStream);
}

AsciiTOP::~AsciiTOP()
{
    if (myInputSurface)  cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface) cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)        cudaStreamDestroy(myStream);
}

void
AsciiTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
    ginfo->cookEveryFrame = false;
    ginfo->cookEveryFrameIfAsked = false;
}

void
AsciiTOP::getInfoPopupString(OP_String* info, void*)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "ASCII v%s", kVersion);
    info->setString(buf);
}

// build the built-in Acerola glyph atlas once, stage for upload
void
AsciiTOP::prepareAtlas()
{
    ascii::GlyphAtlasSpec spec;     // built-in 8px Pixel font
    if (myAtlasValid && spec == myAtlasSpec)
        return;

    ascii::GlyphAtlasResult res;
    std::string err;
    if (!ascii::buildGlyphAtlas(spec, res, &err))
    {
        myError = "Glyph atlas build failed.";
        return;
    }

    myPendingAtlas = std::move(res);
    myPendingSpec  = spec;
    myHasPending   = true;
}

void
AsciiTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    myError = nullptr;
    inputs->enablePar("Version", false);

    if (inputs->getNumInputs() < 1) { myError = "Connect a TOP to the input."; return; }

    const OP_TOPInput* topInput = inputs->getInputTOP(0);
    if (!topInput) { myError = "Input TOP is invalid."; return; }

    const OP_TextureDesc& inDesc = topInput->textureDesc;
    if (inDesc.texDim != OP_TexDim::e2D)
    {
        myError = "Only 2D textures are supported (no 3D / cube / 2D-array).";
        return;
    }
    if (!isSupported8BitRGBA(inDesc.pixelFormat))
    {
        myError = "Input must be 8-bit RGBA/BGRA (BGRA8Fixed or RGBA8Fixed).";
        return;
    }

    TOP_CUDAOutputInfo info;
    info.textureDesc = inDesc;
    info.stream      = myStream;

    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputArrayInfo = topInput->getCUDAArray(acquireInfo, nullptr);

    const OP_CUDAArrayInfo* outputArrayInfo = output->createCUDAArray(info, nullptr);
    if (!outputArrayInfo) { myError = "Failed to create output CUDA array."; return; }

    ascii::Params p;
    p.width  = (int)inDesc.width;
    p.height = (int)inDesc.height;
    p.bgra   = (inDesc.pixelFormat == OP_PixelFormat::BGRA8Fixed);

    int csIdx = std::clamp(inputs->getParInt("Cellsize"), 0, ascii::kNumCellSizes - 1);
    p.cellSize = ascii::kCellSizes[csIdx];

    p.drawEdges     = inputs->getParInt("Drawedges") != 0;
    p.kernelSize    = inputs->getParInt("Kernelsize");
    p.blurSigma     = (float)inputs->getParDouble("Blursigma");
    p.sigmaScale    = (float)inputs->getParDouble("Sigmascale");
    p.dogThreshold  = (float)inputs->getParDouble("Threshold");
    // higher Edge Amount = more edges -> lower internal threshold
    p.edgeThreshold = 1.0f - (float)inputs->getParDouble("Edgeamount");

    p.viewEdges       = inputs->getParInt("Viewedges") != 0;

    p.drawFill    = inputs->getParInt("Drawfill") != 0;
    p.exposure    = (float)inputs->getParDouble("Exposure");
    p.attenuation = (float)inputs->getParDouble("Attenuation");
    p.invert      = inputs->getParInt("Invert") != 0;

    p.fgColor[0] = (float)inputs->getParDouble("Asciicolor", 0);
    p.fgColor[1] = (float)inputs->getParDouble("Asciicolor", 1);
    p.fgColor[2] = (float)inputs->getParDouble("Asciicolor", 2);
    p.fgColor[3] = (float)inputs->getParDouble("Asciicolor", 3);
    p.bgColor[0] = (float)inputs->getParDouble("Backgroundcolor", 0);
    p.bgColor[1] = (float)inputs->getParDouble("Backgroundcolor", 1);
    p.bgColor[2] = (float)inputs->getParDouble("Backgroundcolor", 2);
    p.bgColor[3] = (float)inputs->getParDouble("Backgroundcolor", 3);
    p.tintFromSource = inputs->getParInt("Tintfromsource") != 0;
    p.tintAmount     = (float)inputs->getParDouble("Tintamount");

    p.bypass = inputs->getParInt("Bypass") != 0;

    // build atlas host-side before begin (builds once)
    if (!p.bypass)
        prepareAtlas();

    if (!myContext->beginCUDAOperations(nullptr))
    {
        myError = "beginCUDAOperations() failed.";
        return;
    }

    // upload atlas to device (must be inside begin/end)
    if (myHasPending)
    {
        const char* algoErr = nullptr;
        if (myRenderer.setGlyphAtlas(myPendingAtlas.coverage.data(), myPendingAtlas.glyphPx,
                                     myPendingAtlas.numFill, myStream, &algoErr) == cudaSuccess)
        {
            myAtlasSpec  = myPendingSpec;
            myAtlasValid = true;
        }
        else if (!myError)
        {
            myError = algoErr ? algoErr : "Glyph atlas upload failed.";
        }
        myHasPending = false;
        myPendingAtlas.coverage.clear();
    }

    setupCudaSurface(&myOutputSurface, outputArrayInfo->cudaArray);
    if (inputArrayInfo && inputArrayInfo->cudaArray)
        setupCudaSurface(&myInputSurface, inputArrayInfo->cudaArray);
    else if (myInputSurface)
    {
        cudaDestroySurfaceObject(myInputSurface);
        myInputSurface = 0;
    }

    cudaGetLastError();   // swallow benign sticky errors from surface (re)creation

    const char* algoError = nullptr;
    myRenderer.process(myInputSurface, myOutputSurface, p, myStream, &algoError);
    if (algoError && !myError)
        myError = algoError;

    myContext->endCUDAOperations(nullptr);
}

void
AsciiTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError);
}

static void appendFloat(OP_ParameterManager* m, const char* name, const char* label,
                        const char* page, double def, double lo, double hi,
                        bool clampLo = true, bool clampHi = true)
{
    OP_NumericParameter np(name);
    np.label = label; np.page = page;
    np.defaultValues[0] = def;
    np.minValues[0] = lo;  np.maxValues[0] = hi;
    np.minSliders[0] = lo; np.maxSliders[0] = hi;
    np.clampMins[0] = clampLo; np.clampMaxes[0] = clampHi;
    OP_ParAppendResult r = m->appendFloat(np);
    assert(r == OP_ParAppendResult::Success);
}

static void appendToggle(OP_ParameterManager* m, const char* name, const char* label,
                         const char* page, bool def)
{
    OP_NumericParameter np(name);
    np.label = label; np.page = page; np.defaultValues[0] = def ? 1.0 : 0.0;
    OP_ParAppendResult r = m->appendToggle(np);
    assert(r == OP_ParAppendResult::Success);
}

static void appendColor(OP_ParameterManager* m, const char* name, const char* label,
                        const char* page, double r0, double g0, double b0, double a0)
{
    OP_NumericParameter np(name);
    np.label = label; np.page = page;
    np.defaultValues[0] = r0; np.defaultValues[1] = g0; np.defaultValues[2] = b0; np.defaultValues[3] = a0;
    for (int i = 0; i < 4; ++i) { np.minValues[i] = 0.0; np.maxValues[i] = 1.0;
        np.minSliders[i] = 0.0; np.maxSliders[i] = 1.0; np.clampMins[i] = true; np.clampMaxes[i] = true; }
    OP_ParAppendResult r = m->appendRGBA(np);
    assert(r == OP_ParAppendResult::Success);
}

void
AsciiTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    // ---- ASCII page --------------------------------------------------------
    const char* P = "ASCII";
    appendToggle(manager, "Bypass", "Bypass", P, false);

    {   // cell size (density), pow2: 8 densest, 128 largest
        OP_StringParameter sp("Cellsize");
        sp.label = "Cell Size"; sp.page = P; sp.defaultValue = "C16";
        const char* names[]  = { "C8", "C16", "C32", "C64", "C128" };
        const char* labels[] = { "8 px", "16 px", "32 px", "64 px", "128 px" };
        OP_ParAppendResult r = manager->appendMenu(sp, 5, names, labels);
        assert(r == OP_ParAppendResult::Success);
    }

    // ---- Edges page --------------------------------------------------------
    const char* E = "Edges";
    appendToggle(manager, "Drawedges", "Draw Edges", E, true);
    {   // blur truncation radius (samples)
        OP_NumericParameter np("Kernelsize");
        np.label = "Kernel Size"; np.page = E;
        np.defaultValues[0] = 2;
        np.minValues[0] = 1;  np.maxValues[0] = 8;
        np.minSliders[0] = 1; np.maxSliders[0] = 8;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult r = manager->appendInt(np);
        assert(r == OP_ParAppendResult::Success);
    }
    appendFloat(manager, "Blursigma",  "Blur Strength",   E, 1.25, 0.1, 5.0);
    appendFloat(manager, "Sigmascale", "Deviation Scale", E, 1.25, 1.0, 5.0);
    appendFloat(manager, "Threshold",  "Threshold",       E, 0.005, 0.001, 0.1);
    appendFloat(manager, "Edgeamount", "Edge Amount",     E, 1.0, 0.0, 1.0);

    // debug: output edge map
    appendToggle(manager, "Viewedges", "View Edge Map", E, false);

    // ---- Fill page ---------------------------------------------------------
    const char* F = "Fill";
    appendToggle(manager, "Drawfill",  "Draw Fill", F, true);
    appendFloat(manager, "Exposure",    "Luminance Exposure",    F, 1.0, 0.0, 5.0);
    appendFloat(manager, "Attenuation", "Luminance Attenuation", F, 1.0, 0.0, 5.0);
    appendToggle(manager, "Invert", "Invert", F, false);

    // ---- Color page --------------------------------------------------------
    const char* C = "Color";
    appendColor(manager, "Asciicolor", "ASCII Color", C, 1.0, 1.0, 1.0, 1.0);
    appendColor(manager, "Backgroundcolor", "Background Color", C, 0.0, 0.0, 0.0, 1.0);
    appendToggle(manager, "Tintfromsource", "Tint From Source", C, false);
    appendFloat(manager, "Tintamount", "Tint Amount", C, 1.0, 0.0, 1.0);

    // ---- Version page ------------------------------------------------------
    {
        OP_StringParameter sp("Version");
        sp.label = "Version"; sp.page = "Version"; sp.defaultValue = kVersion;
        OP_ParAppendResult r = manager->appendString(sp);
        assert(r == OP_ParAppendResult::Success);
    }
}
