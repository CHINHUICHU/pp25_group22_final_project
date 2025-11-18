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

echo "Downloading compressed RF data files from Google Drive..."
echo "Source: https://drive.google.com/drive/folders/1F_RahQIT6tuCRNcDl9DlWDhvaqT2ju2c"
echo ""

# Function to download file from Google Drive
download_gdrive() {
    local file_id="$1"
    local output_file="$2"

    echo "Downloading $output_file..."
    curl -L "https://drive.google.com/uc?export=download&id=$file_id" -o "$output_file"

    if [[ ! -f "$output_file" ]]; then
        echo "❌ Failed to download $output_file"
        return 1
    fi
    echo "✓ Downloaded $output_file ($(du -h "$output_file" | cut -f1))"
}

# Download compressed files (you'll need to replace these file IDs with actual ones from your Google Drive)
echo "Note: You need to get the actual Google Drive file IDs and update this script."
echo ""
echo "To get file IDs from Google Drive:"
echo "1. Right-click each .bz2 file in Google Drive"
echo "2. Select 'Get link' and copy the file ID from the URL"
echo "3. Update the file IDs in this script"
echo ""
echo "Manual download instructions:"
echo "1. Visit: https://drive.google.com/drive/folders/1F_RahQIT6tuCRNcDl9DlWDhvaqT2ju2c"
echo "2. Download 01.DAT.bz2, 02.DAT.bz2, 03.DAT.bz2 to testcase/ directory"
echo "3. Run: bunzip2 testcase/*.bz2"

# Uncomment and update file IDs when available:
# download_gdrive "FILE_ID_FOR_01_DAT_BZ2" "testcase/01.DAT.bz2"
# download_gdrive "FILE_ID_FOR_02_DAT_BZ2" "testcase/02.DAT.bz2"
# download_gdrive "FILE_ID_FOR_03_DAT_BZ2" "testcase/03.DAT.bz2"

# Check if compressed files exist (manual download)
if [[ -f "testcase/01.DAT.bz2" && -f "testcase/02.DAT.bz2" && -f "testcase/03.DAT.bz2" ]]; then
    echo ""
    echo "Found compressed files, extracting..."

    bunzip2 testcase/01.DAT.bz2
    bunzip2 testcase/02.DAT.bz2
    bunzip2 testcase/03.DAT.bz2

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
    echo "❌ Compressed files not found. Please download manually:"
    echo "1. Visit the Google Drive link above"
    echo "2. Download all .bz2 files to testcase/ directory"
    echo "3. Run this script again"
fi

echo "=========================================="