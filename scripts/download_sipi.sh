#!/usr/bin/env bash
set -e
mkdir -p data/sample_images
cd data/sample_images

# Example SIPI download (Lena image)
wget -O lena.png https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.04

echo "Downloaded Lena test image into data/sample_images/"
