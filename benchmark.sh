#!/bin/bash
# ============================================
# Benchmark Script for Ultrasound Reconstruction
# Cleans, rebuilds, runs all test cases, records timing, and validates results
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_CASES=("01" "02" "03" "04" "05" "06")
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="${PROJECT_ROOT}/result"
TRUTH_DIR="${PROJECT_ROOT}/truth"

# Default to CUDA version, can be overridden
VERSION="${1:-cuda}"

# Results file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${PROJECT_ROOT}/benchmark_results_${VERSION}_${TIMESTAMP}.txt"

# ============================================
# Helper Functions
# ============================================

print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

log_result() {
    echo "$1" >> "$RESULTS_FILE"
}

# ============================================
# Main Script
# ============================================

print_header "Ultrasound Reconstruction Benchmark"
echo "Version: ${VERSION}"
echo "Timestamp: ${TIMESTAMP}"
echo "Results will be saved to: ${RESULTS_FILE}"

# Initialize results file
log_result "============================================"
log_result "Ultrasound Reconstruction Benchmark Results"
log_result "============================================"
log_result "Version: ${VERSION}"
log_result "Date: $(date)"
log_result "Host: $(hostname)"
log_result ""

# Set directory based on version
if [ "$VERSION" == "cuda" ]; then
    BUILD_DIR="${PROJECT_ROOT}/cuda"
    EXECUTABLE="ultrasound_cuda"
elif [ "$VERSION" == "sequential" ]; then
    BUILD_DIR="${PROJECT_ROOT}/sequential"
    EXECUTABLE="ultrasound"
else
    echo "Usage: $0 [cuda|sequential]"
    exit 1
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    print_error "Build directory not found: ${BUILD_DIR}"
    exit 1
fi

cd "$BUILD_DIR"

# ============================================
# Step 1: Clean
# ============================================
print_header "Step 1: Cleaning build artifacts"
print_info "Running 'make clean'..."
make clean
print_success "Clean completed"
log_result "Build Directory: ${BUILD_DIR}"
log_result ""

# ============================================
# Step 2: Build
# ============================================
print_header "Step 2: Building ${VERSION} version"
print_info "Running 'make'..."

BUILD_START=$(date +%s.%N)
make -j$(nproc) 2>&1 | tee build_log.txt
BUILD_END=$(date +%s.%N)
BUILD_TIME=$(echo "$BUILD_END - $BUILD_START" | bc)

if [ -f "$EXECUTABLE" ]; then
    print_success "Build completed in ${BUILD_TIME}s"
    log_result "Build Time: ${BUILD_TIME}s"
    log_result ""
else
    print_error "Build failed!"
    exit 1
fi

# Also build validate tool if running sequential version
if [ "$VERSION" == "sequential" ]; then
    if [ ! -f "validate" ]; then
        print_error "Validate tool not built!"
        exit 1
    fi
fi

# ============================================
# Step 3: Run Test Cases
# ============================================
print_header "Step 3: Running Test Cases"

# Create result directory
mkdir -p "$RESULT_DIR"

# Arrays to store results
declare -a TIMES
declare -a VALIDATIONS

log_result "============================================"
log_result "Test Case Results"
log_result "============================================"
log_result ""

for CASE in "${TEST_CASES[@]}"; do
    echo ""
    print_info "Running test case ${CASE}..."

    TESTDATA="${PROJECT_ROOT}/testcase/${CASE}.DAT"
    TESTPARAM="${PROJECT_ROOT}/testcase/${CASE}.txt"
    TESTBEAM="beam_${CASE}.dat"
    OUTPUT_PNG="${RESULT_DIR}/image${CASE}.png"
    TRUTH_PNG="${TRUTH_DIR}/${CASE}.png"

    # Check if test data exists
    if [ ! -f "$TESTDATA" ]; then
        print_error "Test data not found: ${TESTDATA}"
        print_info "Run ./download_testdata.sh to download test data"
        TIMES+=("N/A")
        VALIDATIONS+=("SKIPPED")
        log_result "Case ${CASE}: SKIPPED (test data not found)"
        continue
    fi

    # Run with timing
    START_TIME=$(date +%s.%N)

    if ./${EXECUTABLE} "$TESTDATA" "$TESTPARAM" "$TESTBEAM" "$OUTPUT_PNG" 2>&1; then
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        TIMES+=("$ELAPSED")
        print_success "Case ${CASE} completed in ${ELAPSED}s"
    else
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        TIMES+=("${ELAPSED} (FAILED)")
        print_error "Case ${CASE} failed after ${ELAPSED}s"
        VALIDATIONS+=("FAILED")
        log_result "Case ${CASE}: FAILED (execution error)"
        continue
    fi

    # Validate result
    print_info "Validating case ${CASE}..."

    VALIDATE_EXE="${PROJECT_ROOT}/sequential/validate"

    # Build validate if needed
    if [ ! -f "$VALIDATE_EXE" ]; then
        print_info "Building validate tool..."
        (cd "${PROJECT_ROOT}/sequential" && make validate)
    fi

    if [ -f "$TRUTH_PNG" ]; then
        VALIDATION_OUTPUT=$("$VALIDATE_EXE" "$OUTPUT_PNG" "$TRUTH_PNG" 2>&1) || true

        # Check if validation passed (look for PASS, accepted, or similar in output)
        if echo "$VALIDATION_OUTPUT" | grep -qi "pass\|match\|identical\|success\|accepted"; then
            VALIDATIONS+=("PASS")
            print_success "Validation PASSED"
            log_result "Case ${CASE}: ${ELAPSED}s - PASS"
        elif echo "$VALIDATION_OUTPUT" | grep -qi "fail\|mismatch\|different\|error\|rejected"; then
            VALIDATIONS+=("FAIL")
            print_error "Validation FAILED"
            log_result "Case ${CASE}: ${ELAPSED}s - FAIL"
        else
            # If we can't determine, show the output
            VALIDATIONS+=("UNKNOWN")
            print_info "Validation result: ${VALIDATION_OUTPUT}"
            log_result "Case ${CASE}: ${ELAPSED}s - ${VALIDATION_OUTPUT}"
        fi
        echo "  $VALIDATION_OUTPUT"
    else
        VALIDATIONS+=("NO TRUTH")
        print_error "Truth image not found: ${TRUTH_PNG}"
        log_result "Case ${CASE}: ${ELAPSED}s - NO TRUTH FILE"
    fi
done

# ============================================
# Step 4: Summary
# ============================================
print_header "Benchmark Summary"

echo ""
printf "%-10s %-15s %-15s\n" "Case" "Time (s)" "Validation"
printf "%-10s %-15s %-15s\n" "----" "--------" "----------"

TOTAL_TIME=0
for i in "${!TEST_CASES[@]}"; do
    printf "%-10s %-15s %-15s\n" "${TEST_CASES[$i]}" "${TIMES[$i]}" "${VALIDATIONS[$i]}"

    # Sum up times for total (only if numeric)
    if [[ "${TIMES[$i]}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        TOTAL_TIME=$(echo "$TOTAL_TIME + ${TIMES[$i]}" | bc)
    fi
done

echo ""
printf "%-10s %-15s\n" "TOTAL" "${TOTAL_TIME}s"

# Log summary
log_result ""
log_result "============================================"
log_result "Summary"
log_result "============================================"
log_result ""
log_result "$(printf '%-10s %-15s %-15s\n' 'Case' 'Time (s)' 'Validation')"
log_result "$(printf '%-10s %-15s %-15s\n' '----' '--------' '----------')"
for i in "${!TEST_CASES[@]}"; do
    log_result "$(printf '%-10s %-15s %-15s\n' "${TEST_CASES[$i]}" "${TIMES[$i]}" "${VALIDATIONS[$i]}")"
done
log_result ""
log_result "Total Time: ${TOTAL_TIME}s"
log_result ""
log_result "============================================"

print_success "Benchmark complete! Results saved to: ${RESULTS_FILE}"

# Return to project root
cd "$PROJECT_ROOT"
