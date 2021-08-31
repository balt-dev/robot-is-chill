class BabaError(Exception):
    '''Base class for convenient catching'''

class SplittingException(BabaError):
    '''Couldn't split `text_a,b,c` ... somehow
    
    args: cause
    '''

class BadTileProperty(BabaError):
    '''Tried to make a tile a property but it's tooo big

    args: name, size
    '''

class TileNotFound(BabaError):
    '''Unknown tile
    
    args: tile
    '''

class EmptyTile(BabaError):
    '''Blank tiles not allowed'''

class EmptyVariant(BabaError):
    '''Empty variants not allowed
    
    args: tile
    '''

# === Variants ===
class VariantError(BabaError):
    '''Base class for variants
    
    args: tile, variant
    '''

class BadMetaVariant(VariantError):
    '''Too deep
    
    extra args: depth
    '''

class BadPaletteIndex(VariantError):
    '''Not in the palette'''

# TODO: more specific errors for this
class BadTilingVariant(VariantError):
    '''Variant doesn't match tiling
    
    extra args: tiling
    '''

class TileNotText(VariantError):
    '''Can't apply text variants on tiles'''

class BadLetterVariant(VariantError):
    '''Text too long to letterify'''

class UnknownVariant(VariantError):
    '''Not a valid variant'''

# === Custom text ===
class TextGenerationError(BabaError):
    '''Base class for custom text
    
    extra args: text
    '''

class BadLetterStyle(TextGenerationError):
    '''Letter style provided but it's not possible'''

class TooManyLines(TextGenerationError):
    '''Max 1 newline'''

class LeadingTrailingLineBreaks(TextGenerationError):
    '''Can't start or end with newlines'''

class BadCharacter(TextGenerationError):
    '''Invalid character in text
    
    Extra args: mode, char
    '''

class CustomTextTooLong(TextGenerationError):
    '''Can't fit'''