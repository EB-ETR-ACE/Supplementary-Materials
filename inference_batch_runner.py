import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

from inference_config import DEFAULT_OLLAMA_URL, DEFAULT_MODELS, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_BASE_DIR

def run_agent_sequential(
    agent_script: str,
    input_file: Path,
    output_file: Path,
    level: str,
    model_name: str,
    ollama_url: str,
    db_path: Path,
    log_file: Path,
    seed_memory: Path = None,
    python_exe: str = sys.executable
):
    """
    Runs the agent script as a subprocess. This blocks until the process finishes.
    This ensures sequential execution to strictily avoid parallel inference requests.
    """
    cmd = [
        python_exe, agent_script,
        "--input", str(input_file),
        "--output", str(output_file),
        "--level", level,
        "--model-name", model_name,
        "--ollama-url", ollama_url,
        "--db-path", str(db_path),
        "--log-file", str(log_file)
    ]
    
    if seed_memory:
        cmd.extend(["--seed-memory", str(seed_memory)])

    try:
        # capture_output=True captures stdout/stderr so we can log it if needed
        # text=True decodes logs as strings
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error Code {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Sequential Batch Runner for ACE Agents")
    parser.add_argument("--agent-script", default="agent_3node.py", help="Agent script to run")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing .txt files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_BASE_DIR, help="Base directory for results")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to run")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama Server URL")
    parser.add_argument("--level", default="Level 1 (Leichte Sprache)", help="Simplification Level")
    parser.add_argument("--seed-memory", help="Path to seed memory file")
    
    args = parser.parse_args()
    
    base_out = Path(args.output_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found.")
        sys.exit(1)
        
    files = sorted(list(input_dir.glob("*.txt")))
    if not files:
        print(f"No .txt files found in {input_dir}")
        sys.exit(0)

    print(f"--- Starting Sequential Batch Run ---")
    print(f"Agent: {args.agent_script}")
    print(f"Files: {len(files)}")
    print(f"Models: {args.models}")
    print(f"Output: {base_out}")
    print("-------------------------------------")

    all_start_time = time.time()

    for model in args.models:
        model_safe_name = model.replace(":", "_").replace("/", "_")
        run_id = f"{model_safe_name}_{int(time.time())}"
        
        # Structure: output_dir / model_name
        model_out_dir = base_out / model_safe_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # DB and Log specific to this model run to persist memory across files
        db_path = model_out_dir / "agent_memory.sqlite"
        log_path = model_out_dir / "execution.log"
        output_subdir = model_out_dir / "simplified_texts"
        output_subdir.mkdir(exist_ok=True)
        
        print(f"\nProcessing Model: {model} -> {model_out_dir}")
        
        success_count = 0
        
        for f in tqdm(files, desc=f"Running {model_safe_name}"):
            out_file = output_subdir / f.name
            
            # Skip if already exists? (Optional, currently overwriting)
            
            success, logs = run_agent_sequential(
                agent_script=args.agent_script,
                input_file=f,
                output_file=out_file,
                level=args.level,
                model_name=model,
                ollama_url=args.ollama_url,
                db_path=db_path,
                log_file=log_path,
                seed_memory=args.seed_memory
            )
            
            if success:
                success_count += 1
            else:
                # Log failure to console briefly
                tqdm.write(f"Failed file {f.name}: {logs[:200]}...") # Show snippet
                
                # Log full failure to file
                with open(log_path, "a") as lf:
                    lf.write(f"\n[FAILURE] File: {f.name}\n{logs}\n")
        
        print(f"Finished {model}. Success: {success_count}/{len(files)}")

    total_time = time.time() - all_start_time
    print(f"\nAll tasks completed in {total_time:.2f}s.")

if __name__ == "__main__":
    main()
