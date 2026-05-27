#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


CASES_DIR = "cases"
RESULTS_DIR = "results"
SOLVER_TIMEOUT = 125


def discover_cases(filter_pattern=None):
    if not os.path.isdir(CASES_DIR):
        print(f"Error: {CASES_DIR} directory not found", file=sys.stderr)
        sys.exit(1)
    cases = []
    for name in sorted(os.listdir(CASES_DIR)):
        case_dir = os.path.join(CASES_DIR, name)
        if os.path.isdir(case_dir) and os.path.exists(os.path.join(case_dir, "input.json")):
            if filter_pattern is None or filter_pattern in name:
                cases.append(name)
    return cases


def run_case(case_name):
    case_dir = os.path.join(CASES_DIR, case_name)
    input_path = os.path.join(case_dir, "input.json")
    result_dir = os.path.join(RESULTS_DIR, case_name)
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, "output.json")
    temp_path = output_path + ".tmp"

    start = time.time()
    try:
        with open(temp_path, "w") as fout:
            proc = subprocess.run(
                ["python3", "main.py", case_name],
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=SOLVER_TIMEOUT,
            )
        elapsed = time.time() - start
        stderr = proc.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        # Clean up temp file on timeout
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"case": case_name, "status": "TIMEOUT", "elapsed": SOLVER_TIMEOUT}
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"case": case_name, "status": "ERROR", "error": str(e)}

    if proc.returncode != 0:
        # Clean up temp file on non-zero exit
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"case": case_name, "status": "FAIL", "stderr": stderr}

    # Validate temp output is valid JSON before moving
    try:
        with open(temp_path, "r") as f:
            json.load(f)  # Parse to validate it's valid JSON
        # Move temp to final location
        os.replace(temp_path, output_path)
    except (json.JSONDecodeError, IOError) as e:
        # Clean up invalid temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"case": case_name, "status": "FAIL", "stderr": f"Invalid JSON output: {e}\n{stderr}"}

    try:
        cost_line = next(l for l in stderr.splitlines() if l.startswith("Cost:"))
    except StopIteration:
        cost_line = ""

    try:
        validation = subprocess.run(
            ["python3", "validate.py", input_path, output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        val_out = validation.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        val_out = f"validation error: {e}"

    result = {
        "case": case_name,
        "status": "OK" if proc.returncode == 0 else "FAIL",
        "elapsed": round(elapsed, 1),
    }

    for line in val_out.splitlines():
        if line.startswith("Cost (10*HPWL+Area):"):
            result["cost"] = float(line.split(":")[-1].strip())
        elif line.startswith("Overlaps:"):
            result["overlaps"] = line.split(":", 1)[1].strip()
        elif line.startswith("Sym errors:"):
            result["sym_err"] = line.split(":", 1)[1].strip()
        elif line.startswith("Align errors:"):
            result["align_err"] = line.split(":", 1)[1].strip()
        elif line.startswith("Repeat group errors:"):
            result["rg_err"] = line.split(":", 1)[1].strip()

    result["valid"] = (
        result.get("overlaps", "") == "0 []"
        and result.get("sym_err", "") == "[]"
        and result.get("align_err", "") == "[]"
        and result.get("rg_err", "") == "[]"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all test cases")
    parser.add_argument("filter", nargs="?", help="Filter pattern for case names")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1, use 0 for auto/CPU count)")
    args = parser.parse_args()
    
    filter_pattern = args.filter
    jobs = args.jobs
    if jobs == 0:
        jobs = multiprocessing.cpu_count()
    
    cases = discover_cases(filter_pattern)
    if not cases:
        print("No cases found")
        return

    print(f"Running {len(cases)} case(s): {', '.join(cases)}")
    if jobs > 1:
        print(f"Using {jobs} parallel jobs")
    print("-" * 80)
    
    results = []
    start_time = time.time()
    
    if jobs == 1:
        # Sequential execution
        for case in cases:
            print(f"[{case}] running...", end=" ", flush=True)
            r = run_case(case)
            results.append(r)
            cost = r.get("cost", "-")
            valid = "OK" if r.get("valid") else "FAIL"
            elapsed = r.get("elapsed", "-")
            print(f"Cost={cost}, Valid={valid}, Time={elapsed}s")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_case = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    r = future.result()
                    results.append(r)
                    cost = r.get("cost", "-")
                    valid = "OK" if r.get("valid") else "FAIL"
                    elapsed = r.get("elapsed", "-")
                    print(f"[{case}] Cost={cost}, Valid={valid}, Time={elapsed}s", flush=True)
                except Exception as e:
                    print(f"[{case}] ERROR: {e}", flush=True)
                    results.append({"case": case, "status": "ERROR", "error": str(e)})
    
    total_time = time.time() - start_time
    print("-" * 80)
    ok = sum(1 for r in results if r.get("valid"))
    print(f"Summary: {ok}/{len(results)} cases valid in {total_time:.1f}s")

    # Sort results by case name for consistent output
    results.sort(key=lambda x: x["case"])
    
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to results/summary.json")


if __name__ == "__main__":
    main()
