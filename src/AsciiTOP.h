/* ASCII TOP — TD glue.
 * edge-preserving ASCII mosaic (port of Acerola's shader) as one fused CUDA kernel.
 * image processing in AsciiCUDA.{h,cu}; glyphs from GlyphAtlas.{h,cpp}.
 * validated: TouchDesigner 2025.32050, TOP C++ API v12.
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
    // build the built-in glyph atlas once (staged in myPendingAtlas; uploaded in execute())
    void            prepareAtlas();

    const OP_NodeInfo*  myNodeInfo;
    TOP_Context*        myContext;
    cudaStream_t        myStream;

    cudaSurfaceObject_t myInputSurface;
    cudaSurfaceObject_t myOutputSurface;

    ascii::AsciiRenderer myRenderer;

    // glyph atlas cache: rebuild only when spec changes
    ascii::GlyphAtlasSpec   myAtlasSpec;
    bool                    myAtlasValid;
    ascii::GlyphAtlasResult myPendingAtlas;
    ascii::GlyphAtlasSpec   myPendingSpec;
    bool                    myHasPending;

    const char*         myError;
};

#endif // ASCII_TOP_H
