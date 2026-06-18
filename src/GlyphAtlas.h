/* ASCII TOP — host-side glyph atlas builder.
 *
 * Assembles the built-in 8x8 pixel glyphs into a single-channel coverage strip. There is no
 * GDI, no system font, and no external font asset — the glyph bitmaps are embedded in
 * GlyphAtlas.cpp. The strip is built once on the host and uploaded as a device texture.
 *
 * Layout of the produced strip (matches AsciiCUDA.h's glyph order):
 *   [ edge glyphs: | - / \ ]  then  [ fill ramp: numFill glyphs sorted ascending by ink ]
 * each glyph is glyphPx x glyphPx, laid left-to-right; height = glyphPx.
 */
#ifndef ASCII_GLYPH_ATLAS_H
#define ASCII_GLYPH_ATLAS_H

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace ascii {

// Identifies a built atlas; used as the rebuild cache key.
struct GlyphAtlasSpec
{
    char  fontName[128] = "Pixel";
    int   cellSize      = 8;          // glyph rasterization px (power of two)
    bool  crisp         = false;
    bool  bold          = false;
    bool  pixel         = true;       // use the embedded 8x8 pixel font

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
