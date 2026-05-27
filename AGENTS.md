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

## Architecture

**solver.py** uses **Sequence Pair + Simulated Annealing**:
- Sequence pair (alpha, beta permutations) encodes relative positioning → decoded via longest-path DAG → guarantees no overlap for independent boxes
- SA moves: swap, reverse-segment, move-element, axis-adjust, rg-offset-adjust
- Constraint repair after every decode

**Branch strategy** (`long-run`): Single continuous SA run for the entire 120s budget.
- Pro: full cooling schedule, no wasted iterations on re-heating
- Con: can get stuck in local optima, result depends heavily on random seed

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
