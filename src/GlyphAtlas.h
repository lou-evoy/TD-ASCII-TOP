/* ASCII TOP — host-side glyph atlas builder.
 * assembles built-in 8x8 pixel glyphs into a single-channel coverage strip (no GDI/font asset).
 * strip layout (matches AsciiCUDA.h): [edges | - / \] then [fill ramp ascending by ink].
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

// true on success; on failure fills err and caller keeps the previous atlas
bool buildGlyphAtlas(const GlyphAtlasSpec& spec, GlyphAtlasResult& out, std::string* err);

} // namespace ascii

#endif // ASCII_GLYPH_ATLAS_H
