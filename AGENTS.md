# AGENTS.md

## Project
Multi-rule rectangle layout optimization (模拟电路布局). Given box sizes + constraints (symmetry, alignment, repeat groups, nets), find positions minimizing `Cost = 10*HPWL + Area` while satisfying all hard constraints. Python, 120s time limit.

## Commands
```bash
# Run single case
python3 main.py case09-all-constraints

# Validate single output
python3 validate.py cases/{case}/input.json results/{case}/output.json

# Run all cases (batch)
python3 run_all.py

# Run all cases in parallel (e.g., 4 jobs)
python3 run_all.py -j 4

# Run all cases using all CPU cores
python3 run_all.py -j 0

# Run filtered cases with parallel execution
python3 run_all.py -j 4 case0
```

## Test Cases
| Case | Constraint Types | Boxes | Notes |
|------|------------------|-------|-------|
| case00-sample | X symmetry + repeat + align | 32 | Reference sample (baseline) |
| case01-sym-x | X symmetry | 6 | Pure X-axis symmetry |
| case02-sym-y | Y symmetry | 6 | Pure Y-axis symmetry |
| case03-sym-xy | X + Y symmetry | 8 | Independent symmetry groups |
| case04-align | Alignment | 8 | All four alignment types |
| case05-repeat | Repeat groups | 12 | Two repeat groups |
| case06-sym-x-align | X symmetry + alignment | 6 | Constraint interaction |
| case07-sym-x-repeat | X symmetry + repeat | 7 | Symmetry + repeat interaction |
| case08-align-repeat | Alignment + repeat | 6 | Alignment + repeat interaction |
| case09-all-constraints | All constraint types | 12 | Full constraint suite |
| case10-large | X symmetry + repeat + align | 32 | Large-scale test |

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

**solver.py** uses **Constraint Reduction + Simulated Annealing**:
- Linear constraint expressions: each box's (x, y) is a `LinearExpr` over independent variables
- Constraints (symmetry, alignment, repeat) are parsed into linear equations that automatically satisfy hard constraints for any variable assignment
- SA searches over the independent variable space (low-dimensional continuous)
- Objective: `F = 10*HPWL + Area + λ*Overlap`, with λ growing as temperature falls
- Post-processing: gradient-free descent to eliminate residual overlaps

**Key design**:
- `LinearExpr`: linear combinations of variable values, supports arithmetic operations
- `ConstraintReducer`: parses input constraints, builds linear expressions for each box coordinate
- `Optimizer`: SA over variable values with adaptive step sizes per variable

**Branch strategy** (`constraint-reduction`): Constraint-reduction approach
- Pro: hard constraints guaranteed by construction (no constraint repair needed)
- Pro: searches low-dimensional space (fewer variables than boxes)
- Con: complex overlapping constraints on same box can still cause violations (e.g., self-symmetry inside repeat group)

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

Use `run_all.py` to execute all test cases. Supports parallel execution with `-j N` flag (N jobs, or 0 for auto/CPU count). Results saved to `results/summary.json`.
