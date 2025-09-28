#!/usr/bin/env bash
set -e
python3 -m venv venv || true
source venv/bin/activate
pip install -r requirements.txt

# create outputs
mkdir -p outputs/images outputs/signals

# run examples with sample dirs (user-provide data in data/)
python3 src/process_images.py --input_dir data/sample_images --output_dir outputs/images --workers 4 || true
python3 src/process_signals.py --input_dir data/sample_signals --output_dir outputs/signals --workers 4 || true

echo "Done. Check outputs/"
