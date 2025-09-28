#!/usr/bin/env bash
set -e
mkdir -p data/sample_signals
cd data/sample_signals

# Example: download a WAV file (public domain example)
wget -O example.wav https://people.sc.fsu.edu/~jburkardt/data/wav/cantina.wav

echo "Downloaded example wav into data/sample_signals/"
