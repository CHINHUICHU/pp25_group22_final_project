#!/bin/bash

# Download and setup ultrasound test data from Google Drive
# Usage: ./download_testdata.sh

set -e

echo "=========================================="
echo "ULTRASOUND TEST DATA SETUP"
echo "=========================================="

# Create testcase directory if it doesn't exist
mkdir -p testcase

# Check if files already exist
if [[ -f "testcase/01.DAT" && -f "testcase/02.DAT" && -f "testcase/03.DAT" ]]; then
    echo "✓ Test data files already exist. Skipping download."
    echo "Files found:"
    ls -lh testcase/*.DAT
    exit 0
fi

# Google Drive folder URL
FOLDER_URL="https://drive.google.com/drive/folders/1F_RahQIT6tuCRNcDl9DlWDhvaqT2ju2c"

echo "Downloading compressed RF data files from Google Drive..."
echo "Source: $FOLDER_URL"
echo ""

# Check if gdown is available
if ! command -v gdown &> /dev/null && ! python3 -c "import gdown" 2>/dev/null; then
    echo "Installing gdown..."
    pip3 install --user gdown 'beautifulsoup4<4.12' 2>/dev/null || {
        echo "❌ Failed to install gdown. Please install manually:"
        echo "   pip3 install --user gdown 'beautifulsoup4<4.12'"
        echo ""
        echo "Or download manually from: $FOLDER_URL"
        exit 1
    }
fi

# Find gdown executable
GDOWN_CMD=""
if command -v gdown &> /dev/null; then
    GDOWN_CMD="gdown"
elif [[ -f "$HOME/.local/bin/gdown" ]]; then
    GDOWN_CMD="$HOME/.local/bin/gdown"
else
    echo "❌ gdown not found in PATH. Please add ~/.local/bin to PATH or reinstall gdown."
    exit 1
fi

# Download folder using gdown
echo "Using gdown to download folder..."
$GDOWN_CMD --folder "$FOLDER_URL" -O testcase/

# Move files from subdirectory if created
if [[ -d "testcase/pp25_final_testcases" ]]; then
    mv testcase/pp25_final_testcases/*.bz2 testcase/ 2>/dev/null || true
    rmdir testcase/pp25_final_testcases 2>/dev/null || true
fi

# Check if compressed files exist
if [[ -f "testcase/01.DAT.bz2" && -f "testcase/02.DAT.bz2" && -f "testcase/03.DAT.bz2" ]]; then
    echo ""
    echo "Found compressed files, extracting..."

    bunzip2 testcase/01.DAT.bz2
    bunzip2 testcase/02.DAT.bz2
    bunzip2 testcase/03.DAT.bz2
    bunzip2 testcase/04.DAT.bz2
    bunzip2 testcase/05.DAT.bz2
    bunzip2 testcase/06.DAT.bz2

    echo "✓ Extraction complete!"
    echo ""
    echo "Test data files ready:"
    ls -lh testcase/*.DAT

    echo ""
    echo "Verifying file sizes..."
    expected_sizes=(67108864 134217728 67108864)  # 64MB, 128MB, 64MB
    files=(testcase/01.DAT testcase/02.DAT testcase/03.DAT)

    for i in {0..2}; do
        actual_size=$(stat -f%z "${files[$i]}" 2>/dev/null || stat -c%s "${files[$i]}")
        expected_size=${expected_sizes[$i]}

        if [[ $actual_size -eq $expected_size ]]; then
            echo "✓ ${files[$i]}: ${actual_size} bytes (correct)"
        else
            echo "❌ ${files[$i]}: ${actual_size} bytes (expected ${expected_size})"
        fi
    done

else
    echo ""
    echo "❌ Download failed. Compressed files not found."
    echo "Please try downloading manually from: $FOLDER_URL"
    exit 1
fi

echo "=========================================="
