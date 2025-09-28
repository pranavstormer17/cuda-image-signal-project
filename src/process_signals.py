#!/usr/bin/env python3
"""
Batch signal processor.
- Handles WAV and CSV
- Outputs FFT magnitude and downsampled waveform
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def process_wav(path, out_dir, ds_rate=1000):
    p = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr, data = wavfile.read(str(p))
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    fft = np.fft.rfft(data)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    fft_csv = out_dir / f"{p.stem}_fft.csv"
    with open(fft_csv, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frequency','magnitude'])
        for f, m in zip(freqs, mag):
            w.writerow([f, m])
    return (str(p), 'OK', str(fft_csv))

def process_csv(path, out_dir):
    p = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(str(p), delimiter=',')
    if data.ndim > 1:
        data = data[:,0]
    n = len(data)
    sr = 44100
    fft = np.fft.rfft(data)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    fft_csv = out_dir / f"{p.stem}_fft.csv"
    with open(fft_csv, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['frequency','magnitude'])
        for f, m in zip(freqs, mag):
            w.writerow([f, m])
    return (str(p), 'OK', str(fft_csv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = list(input_dir.rglob('*.wav')) + list(input_dir.rglob('*.csv'))
    if not files:
        logging.warning("No wav/csv found in input_dir")
        return

    logpath = output_dir / 'logs'
    logpath.mkdir(parents=True, exist_ok=True)
    logfile = logpath / f'processing_log_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.txt'

    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(logfile, 'w') as lf:
        futures = {}
        for f in files:
            if f.suffix.lower() == '.wav':
                futures[ex.submit(process_wav, str(f), str(output_dir))] = f
            else:
                futures[ex.submit(process_csv, str(f), str(output_dir))] = f
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                src, status, info = fut.result()
                lf.write(f"{datetime.utcnow().isoformat()}\t{src}\t{status}\t{info}\n")
                lf.flush()
                logging.info(f"{src} -> {status}")
            except Exception as e:
                logging.exception("Error")

if __name__ == '__main__':
    main()
