# AGENTS.md

## Project
Multi-rule rectangle layout optimization (模拟电路布局). Given box sizes + constraints (symmetry, alignment, repeat groups, nets), find positions minimizing `Cost = 10*HPWL + Area` while satisfying all hard constraints. Python, 120s time limit.

## Commands
```bash
# Run solver (default case: sample)
python3 main.py

# Run solver with specific case
python3 main.py mycase

# Run solver with custom directory
python3 main.py /path/to/case/directory

# Validate output
python3 validate.py cases/{case}/input.json results/{case}/output.json
```

## Directory Structure
```
cases/
  {case_name}/
    input.json          # 必需：输入数据
    expected.json       # 可选：期望输出（用于对比绘图）

results/
  {case_name}/
    input_boxes.png     # 输入框可视化
    expected_layout.png # 期望输出可视化（如果有 expected.json）
    output_layout.png   # 求解器输出可视化
    output.json         # 求解结果
```

## Architecture

**solver.py** uses **Sequence Pair + Simulated Annealing**:
- Sequence pair (alpha, beta permutations) encodes relative positioning → decoded via longest-path DAG → guarantees no overlap for independent boxes
- SA moves: swap, reverse-segment, move-element, axis-adjust, rg-offset-adjust
- Constraint repair after every decode

**Branch strategy** (`adaptive-restart`): Warm restart SA with adaptive stale detection.
- First run uses full budget; if no improvement for 15s and ≥25s remain, triggers warm restart
- Warm restart perturbs 5-15% of sequence pair positions, uses lower T0 (300-600)
- Pro: escapes local optima, reduces variance across runs
- Con: restart overhead if stale detection is poorly tuned

**Key design**: only independently optimize "independent" boxes (not slaves). Dependent boxes derived from:
- Symmetry pairs: slave = mirror(master) around axis
- Repeat groups: slave = master + offset  
- Self-symmetric: center on axis

**Constraint interaction gotcha**: when master and slave in same repeat group share an alignment constraint, the repeat group offset is forced (e.g., right-align with same width → dx=0). `Solver.forced_offsets` pre-computes these.

## Constraint Application Order (critical)
Must be: symmetry on masters → alignment on non-slaves → repeat groups derive ALL slaves → symmetry on slaves. Wrong order causes constraint violations that look correct but fail validation.

## Current Performance
- Sample: Cost ~35k (reference)
- Solver: Cost ~27-32k (beats reference, all constraints pass)
- ~370k SA iterations in 119s, O(n²) decode per iteration

## Testing
Validate with `validate.py` checks:
- Overlap (bounding box intersection)
- Symmetry axis for all pairs/self-symmetric
- Alignment groups (left/right/top/bottom)  
- Repeat group offset consistency (dx/dy spread < 0.01)
