#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_eval.py — Batch runner for eval_metrics.py.

Iterates through a directory of input files (JSONL or TXT) and runs the evaluation script on each.
Generates corresponding CSV metrics files in the output directory.

Usage:
    python batch_eval.py --input-dir data/v1 --output-dir results/v1
    python batch_eval.py --input-dir data/ --output-dir results/ --recursive
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

# Configuration
EVAL_SCRIPT = "eval_metrics.py"
PYTHON_EXE = sys.executable

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation Runner")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing input files (.jsonl, .txt)")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save output CSVs")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--spacy-model", default="de_core_news_lg", help="SpaCy model to pass to evaluator")
    parser.add_argument("--extension", default="jsonl", help="File extension to look for (default: jsonl)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    # Check for eval script in current directory or strict path
    script_path = Path(__file__).parent / EVAL_SCRIPT
    if not script_path.exists():
        # Fallback to checking typical locations or generic name if in PATH
        script_path = Path(EVAL_SCRIPT)
        if not script_path.exists():
             print(f"Error: Could not find '{EVAL_SCRIPT}' in {Path(__file__).parent}", file=sys.stderr)
             sys.exit(1)

    # Make output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    pattern = f"**/*.{args.extension}" if args.recursive else f"*.{args.extension}"
    files = list(input_dir.glob(pattern))
    
    if not files:
        print(f"No files found with extension '.{args.extension}' in {input_dir}")
        sys.exit(0)

    print(f"Found {len(files)} files to evaluate.")
    
    success_count = 0
    fail_count = 0

    for f in files:
        # Replicate directory structure in output if recursive
        rel_path = f.relative_to(input_dir)
        out_subfile = output_dir / rel_path
        out_csv = out_subfile.with_suffix(".csv")
        out_summary = out_subfile.with_suffix(".json")
        
        # Ensure parent dir exists
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Processing: {f.name} ---")
        cmd = [
            PYTHON_EXE, str(script_path),
            "--input", str(f),
            "--output", str(out_csv),
            "--summary", str(out_summary),
            "--spacy-model", args.spacy_model
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Saved to: {out_csv.name}")
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"❌ Failed to evaluate: {f.name}")
            fail_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            fail_count += 1

    print(f"\nBatch processing complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
