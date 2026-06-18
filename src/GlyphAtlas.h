/* ASCII TOP — host-side glyph atlas builder (GDI).
 *
 * Rasterizes a monospace font into a single-channel coverage strip at the exact cell size,
 * so the GPU samples glyphs 1:1 (crisp at any density, no upscaling). Runs on the CPU only
 * when the font / cell size / style changes — never per frame.
 *
 * Layout of the produced strip (matches AsciiCUDA.h's glyph order):
 *   [ edge glyphs: | - / \ ]  then  [ fill ramp: numFill glyphs sorted ascending by ink ]
 * each glyph is glyphPx x glyphPx, laid left-to-right; height = glyphPx.
 *
 * Windows-only (GDI). This header is host C++ — it is NOT included by the .cu, so it never
 * reaches nvcc. <windows.h> is pulled in by GlyphAtlas.cpp, not here.
 */
#ifndef ASCII_GLYPH_ATLAS_H
#define ASCII_GLYPH_ATLAS_H

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace ascii {

// Everything that affects the rasterized bitmaps. Used as the rebuild cache key.
struct GlyphAtlasSpec
{
    char  fontName[128] = "Pixel";    // monospace family name (GDI face name), or "Pixel"
    int   cellSize      = 8;          // glyph rasterization px (power of two)
    bool  crisp         = false;      // threshold coverage to 1-bit (chunky pixel look)
    bool  bold          = false;
    bool  pixel         = true;       // use the embedded 8x8 pixel font (ignores GDI fields)

    bool operator==(const GlyphAtlasSpec& o) const
    {
        return pixel == o.pixel && cellSize == o.cellSize && crisp == o.crisp &&
               bold == o.bold && std::strncmp(fontName, o.fontName, sizeof(fontName)) == 0;
    }
    bool operator!=(const GlyphAtlasSpec& o) const { return !(*this == o); }
};

struct GlyphAtlasResult
{
    std::vector<uint8_t> coverage;   // (4 + numFill) * glyphPx  x  glyphPx, single channel
    int glyphPx = 0;
    int numFill = 0;
};

// Returns true on success. On failure returns false and (if err) fills a message; the
// caller should fall back to leaving the previous atlas in place.
bool buildGlyphAtlas(const GlyphAtlasSpec& spec, GlyphAtlasResult& out, std::string* err);

} // namespace ascii

#endif // ASCII_GLYPH_ATLAS_H
