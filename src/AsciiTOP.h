/* ASCII TOP — TouchDesigner Custom Operator (C++ TOP, CUDA execute mode).
 *
 * Edge-preserving ASCII / text-mosaic effect. A real-time port of Acerola's ASCII shader
 * (Difference-of-Gaussians edge lines + luminance fill glyphs), rebuilt as a single fused
 * CUDA kernel so it scales to huge input resolutions.
 *
 * Thin SDK-facing glue. All image processing lives in AsciiCUDA.{h,cu}; the glyph bitmaps
 * are rasterized on the host in GlyphAtlas.{h,cpp} (GDI) and uploaded as a texture.
 *
 * Validated against TouchDesigner 2025.32050, TOP C++ API version 12.
 */
#ifndef ASCII_TOP_H
#define ASCII_TOP_H

#include "TOP_CPlusPlusBase.h"
#include "cuda_runtime.h"
#include "AsciiCUDA.h"
#include "GlyphAtlas.h"

using namespace TD;

class AsciiTOP : public TOP_CPlusPlusBase
{
public:
    AsciiTOP(const OP_NodeInfo* info, TOP_Context* context);
    virtual ~AsciiTOP();

    virtual void    getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
    virtual void    execute(TOP_Output*, const OP_Inputs*, void* reserved1) override;

    virtual void    getErrorString(OP_String* error, void* reserved1) override;
    virtual void    getInfoPopupString(OP_String* info, void* reserved1) override;

    virtual void    setupParameters(OP_ParameterManager* manager, void* reserved1) override;

private:
    // Build the (fixed, built-in) pixel glyph atlas once, staging it in myPendingAtlas. The
    // device upload happens in execute() inside the begin/end block.
    void            prepareAtlas();

    const OP_NodeInfo*  myNodeInfo;
    TOP_Context*        myContext;
    cudaStream_t        myStream;

    cudaSurfaceObject_t myInputSurface;
    cudaSurfaceObject_t myOutputSurface;

    ascii::AsciiRenderer myRenderer;

    // Glyph atlas cache: rebuild only when the spec changes. A freshly rasterized atlas is
    // staged in myPendingAtlas (host) and uploaded to the device inside the begin/end block.
    ascii::GlyphAtlasSpec   myAtlasSpec;
    bool                    myAtlasValid;
    ascii::GlyphAtlasResult myPendingAtlas;
    ascii::GlyphAtlasSpec   myPendingSpec;
    bool                    myHasPending;

    const char*         myError;
};

#endif // ASCII_TOP_H
