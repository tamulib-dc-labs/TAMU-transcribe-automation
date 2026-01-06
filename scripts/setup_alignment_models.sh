#!/bin/bash
# Setup script to ensure alignment models are properly organized for offline use
# Run this on the LOGIN NODE after downloading models

CACHE_DIR="/scratch/group/tamu_libr_dc/cache/huggingface"

echo "=================================="
echo "Alignment Model Setup for Offline Mode"
echo "=================================="
echo "Cache directory: $CACHE_DIR"
echo ""

# Create checkpoints directory
CHECKPOINTS_DIR="$CACHE_DIR/checkpoints"
mkdir -p "$CHECKPOINTS_DIR"
echo "✓ Created checkpoints directory"

# Check if alignment models exist at root level
if ls "$CACHE_DIR"/wav2vec2_*.pth >/dev/null 2>&1 || ls "$CACHE_DIR"/wav2vec2_*.pt >/dev/null 2>&1; then
    echo ""
    echo "Found alignment models at root level, copying to checkpoints/..."
    
    # Copy .pth files
    for file in "$CACHE_DIR"/wav2vec2_*.pth; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ ! -f "$CHECKPOINTS_DIR/$filename" ]; then
                cp "$file" "$CHECKPOINTS_DIR/"
                echo "  ✓ Copied $filename"
            else
                echo "  - $filename (already exists)"
            fi
        fi
    done
    
    # Copy .pt files
    for file in "$CACHE_DIR"/wav2vec2_*.pt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [ ! -f "$CHECKPOINTS_DIR/$filename" ]; then
                cp "$file" "$CHECKPOINTS_DIR/"
                echo "  ✓ Copied $filename"
            else
                echo "  - $filename (already exists)"
            fi
        fi
    done
    
    echo ""
    echo "✓ Alignment models ready for offline use!"
else
    echo ""
    echo "⚠ WARNING: No alignment models found at $CACHE_DIR"
    echo ""
    echo "To download alignment models, run on LOGIN NODE:"
    echo "  export HF_HOME=$CACHE_DIR"
    echo "  python << 'EOF'"
    echo "import whisperx"
    echo "align_model, _ = whisperx.load_align_model('en', 'cpu')"
    echo "print('✓ Downloaded')"
    echo "EOF"
fi

echo ""
echo "=================================="
echo "Verification"
echo "=================================="
echo "Alignment models in checkpoints/:"
ls -lh "$CHECKPOINTS_DIR"/ | grep wav2vec2 || echo "  (none found)"
echo ""
echo "Setup complete!"
