#!/usr/bin/env python3
"""
Batch image processor.
- Resizes images
- Converts to grayscale
- Saves histogram CSV
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import csv
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def process_one(path, out_dir, max_dim=1024):
    p = Path(path)
    stem = p.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("Image read returned None")
        h, w = img.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc_path = out_dir / f"{stem}_gray.png"
        _, encoded = cv2.imencode('.png', gray)
        encoded.tofile(str(proc_path))
        # histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten().astype(int)
        hist_csv = out_dir / f"{stem}_hist.csv"
        with open(hist_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['bin', 'count'])
            for i, c in enumerate(hist):
                writer.writerow([i, int(c)])
        return (str(p), 'OK', str(proc_path))
    except Exception as e:
        return (str(p), 'ERROR', str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_dim', type=int, default=1024)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f'processing_log_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.txt'

    image_files = []
    for ext in ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'):
        image_files.extend(input_dir.rglob(ext))

    if len(image_files) == 0:
        logging.warning("No image files found in input_dir")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(logfile, 'w') as lf:
        futures = { ex.submit(process_one, str(p), str(output_dir), args.max_dim): p for p in image_files }
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                src_path, status, info = fut.result()
                t = datetime.utcnow().isoformat()
                lf.write(f"{t}\t{src_path}\t{status}\t{info}\n")
                lf.flush()
                logging.info(f"{src_path} -> {status}")
            except Exception as e:
                logging.exception("Processing error")

if __name__ == '__main__':
    main()
