#!/bin/bash
# Pull all Apptainer/Singularity images for the SWE-smith training set.
# Resume-safe: already-downloaded .sif files are skipped.
#
# Usage: bash pull_swe_smith_images.sh

set -e

# Resolve paths relative to this script so it runs from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARQUET_FILE="${PARQUET_FILE:-$SCRIPT_DIR/../../Data/Agentic/swe_smith_train.parquet}"
IMAGE_DIR="${IMAGE_DIR:-$SCRIPT_DIR/../../Data/Agentic/swe_smith_images}"

mkdir -p "$IMAGE_DIR"

# Extract the unique image names from the parquet
python -c "
import pandas as pd
df = pd.read_parquet('$PARQUET_FILE')
for image in sorted(df['image_name'].dropna().unique()):
    print(str(image).lower())
" | sort -u > "$IMAGE_DIR/image_list.txt"

TOTAL=$(wc -l < "$IMAGE_DIR/image_list.txt")
echo "Total unique images to pull: $TOTAL"

COUNT=0
while IFS= read -r image; do
    COUNT=$((COUNT + 1))
    # Convert image name to a safe filename for the SIF file
    sif_name=$(echo "$image" | sed 's|docker.io/||; s|/|_|g; s|:|-|g').sif

    if [ -f "$IMAGE_DIR/$sif_name" ]; then
        echo "[$COUNT/$TOTAL] SKIP (exists): $sif_name"
        continue
    fi

    echo "[$COUNT/$TOTAL] Pulling: $image -> $sif_name"
    apptainer pull "$IMAGE_DIR/$sif_name" "docker://${image#docker.io/}"
done < "$IMAGE_DIR/image_list.txt"

echo "Done! Images saved to $IMAGE_DIR"
