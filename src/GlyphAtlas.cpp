/* ASCII TOP — built-in pixel glyph atlas. See GlyphAtlas.h.
 *
 * Acerola's exact 8x8 glyphs, decoded straight from AcerolaFX's edgesASCII.png / fillASCII.png
 * and embedded below, assembled into a single coverage strip:
 *   [ edges: | - / \ ]  then  [ fill: empty . : C O P 0 ? @ block ]  (dark -> bright)
 * Genuinely pixel-perfect (designed on the grid) and a 1:1 match to his shader. No system
 * fonts, no GDI — the atlas is fixed and built once.
 */
#include "GlyphAtlas.h"
#include "AsciiCUDA.h"   // kNumEdgeGlyphs (shared glyph-slot order)

namespace ascii {

// Row 0 = top; bit i (LSB = leftmost) = column i.
struct PixelGlyph { unsigned char rows[8]; };

// Edge glyphs in slot order (V, H, /, \) = Acerola edge slots 1,2,3,4.
static const PixelGlyph kPixelEdges[ascii::kNumEdgeGlyphs] = {
    {{0x00,0x08,0x08,0x08,0x08,0x08,0x08,0x00}}, // |  vertical
    {{0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00}}, // -  horizontal
    {{0x00,0x20,0x10,0x10,0x08,0x08,0x04,0x00}}, // /
    {{0x00,0x04,0x08,0x08,0x10,0x10,0x20,0x00}}, // backslash
};
// Acerola's 10-level fill ramp (fillASCII.png), dark -> bright.
static const PixelGlyph kPixelFill[] = {
    {{0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}}, // (empty)
    {{0x00,0x00,0x00,0x00,0x00,0x00,0x10,0x00}}, // .
    {{0x00,0x00,0x00,0x10,0x00,0x10,0x10,0x00}}, // :
    {{0x00,0x00,0x18,0x24,0x04,0x24,0x18,0x00}}, // C
    {{0x00,0x00,0x0C,0x12,0x12,0x12,0x0C,0x00}}, // O
    {{0x00,0x1C,0x24,0x24,0x1C,0x04,0x04,0x00}}, // P
    {{0x00,0x18,0x24,0x24,0x24,0x24,0x18,0x00}}, // 0
    {{0x00,0x18,0x24,0x30,0x08,0x00,0x08,0x00}}, // ?
    {{0x00,0x38,0x44,0x74,0x24,0x08,0x30,0x00}}, // @
    {{0x00,0x7E,0x7E,0x7E,0x7E,0x7E,0x7E,0x00}}, // block
};

static void pixelGlyphToCoverage(const PixelGlyph& g, std::vector<uint8_t>& dst)
{
    dst.assign(64, 0);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i)
            dst[(size_t)j * 8 + i] = ((g.rows[j] >> i) & 1) ? 255 : 0;
}

bool buildGlyphAtlas(const GlyphAtlasSpec& /*spec*/, GlyphAtlasResult& out, std::string* /*err*/)
{
    const int px      = 8;
    const int numFill = (int)(sizeof(kPixelFill) / sizeof(kPixelFill[0]));
    const int total   = kNumEdgeGlyphs + numFill;
    const int stripW  = total * px;

    out.glyphPx = px;
    out.numFill = numFill;
    out.coverage.assign((size_t)stripW * px, 0);

    std::vector<uint8_t> g;
    auto blit = [&](int slot, const PixelGlyph& pg) {
        pixelGlyphToCoverage(pg, g);
        for (int j = 0; j < px; ++j)
            for (int i = 0; i < px; ++i)
                out.coverage[(size_t)j * stripW + slot * px + i] = g[(size_t)j * px + i];
    };

    for (int e = 0; e < kNumEdgeGlyphs; ++e) blit(e, kPixelEdges[e]);
    for (int f = 0; f < numFill; ++f)        blit(kNumEdgeGlyphs + f, kPixelFill[f]);

    return true;
}

} // namespace ascii
