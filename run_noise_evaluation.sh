#!/bin/bash

# Noise Evaluation Script for Voice Conversion
# This script runs the comprehensive noise evaluation

# Configuration
GEN_ROOT="/home/rufael/Projects/voice-anonymization/exps/knn-vc/generated/noise"  # Directory with generated audio
TGT_ROOT="/home/rufael/Projects/voice-anonymization/data/data-seed-vc-test"          # Directory with target/reference audio
TXT_PATH="/home/rufael/Projects/voice-anonymization/data-16k/ArVoice/test.txt"                   # Path to transcript file
OUTPUT_DIR="/home/rufael/Projects/voice-anonymization/exps/knn-vc/results/noise" # Output directory for results
DEVICE="cuda"                         # Device to use (cuda/cpu)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=== Noise Impact Evaluation for Voice Conversion ==="
echo "Generated audio directory: $GEN_ROOT"
echo "Target audio directory: $TGT_ROOT"
echo "Transcript file: $TXT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Check if generated audio exists
if [ ! -d "$GEN_ROOT" ]; then
    echo "Error: Generated audio directory '$GEN_ROOT' not found!"
    echo "Please run the decode-noise.py script first to generate noisy audio."
    exit 1
fi

# Check if transcript file exists
if [ ! -f "$TXT_PATH" ]; then
    echo "Warning: Transcript file '$TXT_PATH' not found!"
    echo "WER calculation will be skipped."
    TXT_PATH=""
fi

# Run evaluation
echo "Starting evaluation..."
python evaluate_noise.py \
    --gen_root "$GEN_ROOT" \
    --tgt_root "$TGT_ROOT" \
    --txt_path "$TXT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --create_plots

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to: $OUTPUT_DIR/plots/"
echo "Summary table: $OUTPUT_DIR/plots/summary_table.csv"
echo "All results: $OUTPUT_DIR/all_results.csv" 