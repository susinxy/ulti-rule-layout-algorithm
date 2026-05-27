import math, random, time, sys
from typing import Dict, List, Tuple, Optional

EPS = 1e-9


class LinearExpr:
    """Linear expression: value = const + sum(coeffs[v] * v for v in coeffs)."""
    __slots__ = ('const', 'coeffs')

    def __init__(self, const: float = 0.0, coeffs: Optional[Dict[int, float]] = None):
        self.const = float(const)
        self.coeffs = dict(coeffs) if coeffs else {}

    def __add__(self, other):
        if isinstance(other, (int, float)):
            e = LinearExpr(self.const + float(other), dict(self.coeffs))
            return e
        result_coeffs = dict(self.coeffs)
        for v, c in other.coeffs.items():
            result_coeffs[v] = result_coeffs.get(v, 0.0) + c
        return LinearExpr(self.const + other.const, result_coeffs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return LinearExpr(self.const - float(other), dict(self.coeffs))
        result_coeffs = dict(self.coeffs)
        for v, c in other.coeffs.items():
            result_coeffs[v] = result_coeffs.get(v, 0.0) - c
        return LinearExpr(self.const - other.const, result_coeffs)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return LinearExpr(-self.const, {v: -c for v, c in self.coeffs.items()})

    def __mul__(self, scalar):
        scalar = float(scalar)
        return LinearExpr(self.const * scalar, {v: c * scalar for v, c in self.coeffs.items()})

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1.0 / float(scalar))

    def evaluate(self, values: Dict[int, float]) -> float:
        """Evaluate the expression given values for all variables."""
        total = self.const
        for v, c in self.coeffs.items():
            total += c * values.get(v, 0.0)
        return total

    def vars(self):
        return set(self.coeffs.keys())

    def is_constant(self):
        return len(self.coeffs) == 0


class ConstraintReducer:
    """
    Reduces hard constraints (symmetry, alignment, repeat groups) into linear
    expressions over independent variables.

    After reduction, each box's (x, y) is a LinearExpr: x_i = sum(c_v * v) + k.
    All hard constraints are automatically satisfied by any assignment of the
    independent variables.
    """

    def __init__(self, data):
        self.data = data
        self.box_sizes = data['box_size']
        self.n = len(self.box_sizes)
        # 0-indexed widths and heights
        self.w = [float(s[0]) for s in self.box_sizes]
        self.h = [float(s[1]) for s in self.box_sizes]

        # Each box starts with its own free x, y variables
        self.num_vars = 0
        self.x_expr: List[LinearExpr] = [None] * self.n
        self.y_expr: List[LinearExpr] = [None] * self.n
        for i in range(self.n):
            self.x_expr[i] = self._fresh_var()
            self.y_expr[i] = self._fresh_var()

        # Track which var_ids correspond to axes / offsets for initial-state heuristic
        self.axis_vars: List[int] = []  # all axis vars (x and y)
        self.axis_x_vars: List[int] = []  # x-axis symmetry vars only
        self.axis_y_vars: List[int] = []  # y-axis symmetry vars only
        self.offset_vars: List[Tuple[int, int]] = []  # (dx_var, dy_var) pairs

        # Process constraints in order: symmetry -> alignment -> repeat groups
        self._process_symmetry()
        self._process_alignment()
        self._process_repeat_groups()

    def _fresh_var(self) -> LinearExpr:
        v = self.num_vars
        self.num_vars += 1
        return LinearExpr(0.0, {v: 1.0})

    def _alloc_var_id(self) -> int:
        v = self.num_vars
        self.num_vars += 1
        return v

    def _process_symmetry(self):
        # X-axis symmetry groups
        for group in self.data.get('symmetry_x', []):
            axis_id = self._alloc_var_id()
            self.axis_vars.append(axis_id)
            self.axis_x_vars.append(axis_id)
            axis_expr = LinearExpr(0.0, {axis_id: 1.0})

            # Pairs: (a, b) - b is derived from a and axis
            for pair in group.get('symmetry_pair', []):
                a, b = pair[0] - 1, pair[1] - 1
                # center_x(b) = 2 * axis - center_x(a)
                # b.x + w_b/2 = 2*axis - (a.x + w_a/2)
                # b.x = 2*axis - a.x - w_a/2 - w_b/2
                self.x_expr[b] = 2 * axis_expr - self.x_expr[a] - (self.w[a] + self.w[b]) / 2.0
                # Pair boxes share the same y-center: y_b + h_b/2 = y_a + h_a/2
                self.y_expr[b] = self.y_expr[a] + (self.h[a] - self.h[b]) / 2.0

            # Self-symmetric: center on the axis
            for s in group.get('self_symmetry', []):
                s -= 1
                # x + w/2 = axis  -->  x = axis - w/2
                self.x_expr[s] = axis_expr - self.w[s] / 2.0
                # y is unchanged (still free var from init)

        # Y-axis symmetry groups
        for group in self.data.get('symmetry_y', []):
            axis_id = self._alloc_var_id()
            self.axis_vars.append(axis_id)
            self.axis_y_vars.append(axis_id)
            axis_expr = LinearExpr(0.0, {axis_id: 1.0})

            for pair in group.get('symmetry_pair', []):
                a, b = pair[0] - 1, pair[1] - 1
                # y_b + h_b/2 = 2*axis - (y_a + h_a/2)
                self.y_expr[b] = 2 * axis_expr - self.y_expr[a] - (self.h[a] + self.h[b]) / 2.0
                # Same x-center
                self.x_expr[b] = self.x_expr[a] + (self.w[a] - self.w[b]) / 2.0

            for s in group.get('self_symmetry', []):
                s -= 1
                self.y_expr[s] = axis_expr - self.h[s] / 2.0

    def _process_alignment(self):
        align = self.data.get('align', {})

        # left: all boxes share the same x coordinate
        for grp in align.get('left', []):
            base = grp[0] - 1
            for b in grp[1:]:
                b -= 1
                self.x_expr[b] = self.x_expr[base]

        # right: all boxes share the same right edge (x + w)
        # x_b + w_b = x_base + w_base  -->  x_b = x_base + w_base - w_b
        for grp in align.get('right', []):
            base = grp[0] - 1
            for b in grp[1:]:
                b -= 1
                self.x_expr[b] = self.x_expr[base] + self.w[base] - self.w[b]

        # top: all boxes share the same top edge (y + h)
        for grp in align.get('top', []):
            base = grp[0] - 1
            for b in grp[1:]:
                b -= 1
                self.y_expr[b] = self.y_expr[base] + self.h[base] - self.h[b]

        # bottom: all boxes share the same y coordinate
        for grp in align.get('bottom', []):
            base = grp[0] - 1
            for b in grp[1:]:
                b -= 1
                self.y_expr[b] = self.y_expr[base]

    def _process_repeat_groups(self):
        for rg in self.data.get('repeat_groups', []):
            groups = rg['groups']
            if len(groups) < 2:
                continue
            master = [b - 1 for b in groups[0]]

            # One (dx, dy) offset per additional group
            for slave_group in groups[1:]:
                slave = [b - 1 for b in slave_group]
                dx_id = self._alloc_var_id()
                dy_id = self._alloc_var_id()
                self.offset_vars.append((dx_id, dy_id))
                dx_expr = LinearExpr(0.0, {dx_id: 1.0})
                dy_expr = LinearExpr(0.0, {dy_id: 1.0})
                # slave_i = master_i + offset
                for m, s in zip(master, slave):
                    self.x_expr[s] = self.x_expr[m] + dx_expr
                    self.y_expr[s] = self.y_expr[m] + dy_expr

    def evaluate(self, values: Dict[int, float]) -> Dict[int, Tuple[float, float]]:
        """Given var values, compute each box's (x, y) left-bottom corner.
        Returns 1-indexed positions to match the expected output format."""
        return {
            i + 1: (self.x_expr[i].evaluate(values), self.y_expr[i].evaluate(values))
            for i in range(self.n)
        }

    def var_box_indices(self, v: int) -> List[int]:
        """Indices of boxes whose x or y expression uses var v. Useful for
        tuning per-variable step sizes."""
        indices: List[int] = []
        for i in range(self.n):
            if v in self.x_expr[i].coeffs or v in self.y_expr[i].coeffs:
                indices.append(i)
        return indices


class Optimizer:
    """Simulated annealing over the independent variable space produced by
    ConstraintReducer. Overlap is handled via a penalty term that grows as the
    temperature falls."""

    def __init__(self, reducer: ConstraintReducer):
        self.reducer = reducer
        self.data = reducer.data
        # Keep nets as 1-indexed to match positions dict
        self.nets = reducer.data.get('nets', [])
        self.widths = reducer.w
        self.heights = reducer.h
        self.n = reducer.n
        self.num_vars = reducer.num_vars

        # Per-variable step scale: max of widths and heights of boxes that
        # depend on that var. Gives sensible move magnitudes.
        self.step_scales = []
        for v in range(self.num_vars):
            max_dim = 1.0
            for i in self.reducer.var_box_indices(v):
                max_dim = max(max_dim, self.widths[i], self.heights[i])
            self.step_scales.append(max_dim)

    def _evaluate(self, state: Dict[int, float]) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
        """Returns (cost_no_overlap, overlap_area, positions dict)."""
        positions = self.reducer.evaluate(state)

        # HPWL - positions are 1-indexed, nets use 1-indexed box IDs
        hpwl = 0.0
        for net in self.nets:
            if not net:
                continue
            cxs = [positions[box_id][0] + self.widths[box_id - 1] / 2 for box_id in net]
            cys = [positions[box_id][1] + self.heights[box_id - 1] / 2 for box_id in net]
            hpwl += (max(cxs) - min(cxs)) + (max(cys) - min(cys))

        # Bounding box - positions are 1-indexed
        min_x = min(positions[i][0] for i in range(1, self.n + 1))
        min_y = min(positions[i][1] for i in range(1, self.n + 1))
        max_x = max(positions[i][0] + self.widths[i - 1] for i in range(1, self.n + 1))
        max_y = max(positions[i][1] + self.heights[i - 1] for i in range(1, self.n + 1))
        area = (max_x - min_x) * (max_y - min_y)

        # Overlap - positions are 1-indexed
        overlap = 0.0
        for i in range(1, self.n + 1):
            xi, yi = positions[i]
            xi2, yi2 = xi + self.widths[i - 1], yi + self.heights[i - 1]
            for j in range(i + 1, self.n + 1):
                xj, yj = positions[j]
                ox = min(xi2, xj + self.widths[j - 1]) - max(xi, xj)
                oy = min(yi2, yj + self.heights[j - 1]) - max(yi, yj)
                if ox > 0 and oy > 0:
                    overlap += ox * oy

        return 10.0 * hpwl + area, overlap, positions

    def _initial_state(self) -> Dict[int, float]:
        """Heuristic initial state: spread boxes on a grid so that there is
        little or no overlap from the start, then pick variable values that
        (for boxes whose x/y are a single fresh var) reproduce the grid."""
        state: Dict[int, float] = {v: 0.0 for v in range(self.num_vars)}

        # Grid layout of boxes by size
        max_row_width = (sum(self.widths[i] * self.heights[i] for i in range(self.n))) ** 0.5 * 1.5
        order = sorted(range(self.n), key=lambda i: -(self.widths[i] * self.heights[i]))
        initial_pos: List[Tuple[float, float]] = [(0.0, 0.0)] * self.n
        x_cursor, y_cursor, row_height = 0.0, 0.0, 0.0
        for i in order:
            if x_cursor + self.widths[i] > max_row_width and x_cursor > 0:
                x_cursor = 0.0
                y_cursor += row_height
                row_height = 0.0
            initial_pos[i] = (x_cursor, y_cursor)
            x_cursor += self.widths[i]
            row_height = max(row_height, self.heights[i])

        # Compute bounding box center
        all_x = [p[0] + self.widths[i] / 2 for i, p in enumerate(initial_pos)]
        all_y = [p[1] + self.heights[i] / 2 for i, p in enumerate(initial_pos)]
        bbox_cx = (min(all_x) + max(all_x)) / 2
        bbox_cy = (min(all_y) + max(all_y)) / 2

        # For each free box var (x[i] = single fresh var, y[i] = single fresh var),
        # try to set that var to the grid position.
        for i in range(self.n):
            x_expr = self.reducer.x_expr[i]
            y_expr = self.reducer.y_expr[i]
            target_x, target_y = initial_pos[i]

            # x
            if len(x_expr.coeffs) == 1:
                v, coeff = next(iter(x_expr.coeffs.items()))
                if abs(coeff - 1.0) < 1e-9:
                    state[v] = target_x - x_expr.const

            # y
            if len(y_expr.coeffs) == 1:
                v, coeff = next(iter(y_expr.coeffs.items()))
                if abs(coeff - 1.0) < 1e-9:
                    state[v] = target_y - y_expr.const

        # Set axis variables: axis_x near bbox_cx, axis_y near bbox_cy
        # Axis variables appear with coeff 2.0 in expressions like: x = 2*axis - x₀ - w
        for axis_id in self.reducer.axis_x_vars:
            state[axis_id] = bbox_cx

        for axis_id in self.reducer.axis_y_vars:
            state[axis_id] = bbox_cy

        # Offsets: set to a moderate spacing so slave groups don't stack on masters
        avg_w = sum(self.widths) / self.n
        avg_h = sum(self.heights) / self.n
        for dx_id, dy_id in self.reducer.offset_vars:
            state[dx_id] = avg_w * 1.5
            state[dy_id] = avg_h * 1.5

        return state

    def _postprocess(self, state: Dict[int, float], max_iters: int = 500) -> Dict[int, float]:
        """Simple gradient-free descent targeting overlap elimination. Uses a
        1D line search along each variable direction, repeating until no
        overlap remains or iterations exhausted."""
        best = dict(state)
        cur_cost, cur_overlap, _ = self._evaluate(best)

        for _ in range(max_iters):
            if cur_overlap < 1e-6:
                return best

            improved = False
            for v in range(self.num_vars):
                scale = self.step_scales[v] * 0.1
                for direction in (+1.0, -1.0):
                    trial = dict(best)
                    trial[v] = best[v] + direction * scale
                    t_cost, t_overlap, _ = self._evaluate(trial)
                    if t_overlap < cur_overlap - 1e-9 or (t_overlap < 1e-6 and t_cost < cur_cost):
                        best = trial
                        cur_cost, cur_overlap = t_cost, t_overlap
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Reduce step sizes and retry once
                for v in range(self.num_vars):
                    self.step_scales[v] *= 0.7
                # If step_scales collapsed, give up
                if max(self.step_scales) < 1e-3:
                    break

        return best

    def solve(self, time_limit: float = 120.0) -> Tuple[Optional[Dict[int, Tuple[float, float]]], float]:
        start = time.time()

        # Try a few random-restart SA runs within the budget
        best_positions: Optional[Dict[int, Tuple[float, float]]] = None
        best_cost = float('inf')
        best_state: Optional[Dict[int, float]] = None
        best_overlap = float('inf')

        # Tuning
        T_start = 5000.0
        T_min = 1.0
        cooling = 0.9997
        # Lambda grows as T falls to progressively enforce no-overlap
        lambda_base = 10.0

        run_count = 0
        stale_threshold = 20.0  # seconds without improvement triggers restart
        last_global_improve = start

        while time.time() - start < time_limit - 2.0:
            run_count += 1
            is_first = (run_count == 1)

            if is_first or best_state is None:
                state = self._initial_state()
                T = T_start
            else:
                # Warm restart: perturb the best state
                state = dict(best_state)
                perturb_frac = random.uniform(0.05, 0.20)
                n_perturb = max(1, int(self.num_vars * perturb_frac))
                for _ in range(n_perturb):
                    v = random.randint(0, self.num_vars - 1)
                    state[v] = state[v] + random.gauss(0, self.step_scales[v] * 0.5)
                T = random.uniform(500, 1500)
                cooling = 0.9999

            cur_cost, cur_overlap, cur_pos = self._evaluate(state)
            cur_score = cur_cost + (lambda_base / max(T, T_min)) * cur_overlap

            local_best_state = dict(state)
            local_best_cost = cur_cost if cur_overlap < 1e-6 else float('inf')
            local_best_overlap = cur_overlap

            chunk_last_improve = time.time()
            moves = 0

            while time.time() - start < time_limit - 1.0:
                remaining_chunk = time.time() - chunk_last_improve
                if not is_first and remaining_chunk > 25.0:
                    break

                v = random.randint(0, self.num_vars - 1)
                step = random.gauss(0, self.step_scales[v] * max(T / T_start, 0.01))
                old_val = state[v]
                state[v] = old_val + step

                new_cost, new_overlap, new_pos = self._evaluate(state)
                lam = lambda_base / max(T, T_min)
                new_score = new_cost + lam * new_overlap
                delta = new_score - cur_score

                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-3)):
                    cur_cost, cur_overlap = new_cost, new_overlap
                    cur_score = new_score
                    if cur_overlap < local_best_overlap:
                        local_best_cost = cur_cost if cur_overlap < 1e-6 else float('inf')
                        local_best_overlap = cur_overlap
                        local_best_state = dict(state)
                        chunk_last_improve = time.time()
                else:
                    state[v] = old_val

                T = max(T * cooling, T_min)
                moves += 1

            # Post-process the local best if overlaps remain
            if local_best_overlap > 1e-6:
                pp_state = self._postprocess(local_best_state)
                pp_cost, pp_overlap, pp_pos = self._evaluate(pp_state)
                if pp_overlap < local_best_overlap:
                    local_best_state = pp_state
                    local_best_cost = pp_cost if pp_overlap < 1e-6 else local_best_cost
                    local_best_overlap = pp_overlap

            elapsed = time.time() - start
            print(
                f"Run {run_count}: moves={moves}, local_cost={local_best_cost:.2f}, "
                f"local_overlap={local_best_overlap:.4f}, "
                f"global_best={best_cost:.2f}, {elapsed:.1f}s",
                file=sys.stderr,
            )

            if local_best_cost < best_cost and local_best_overlap < 1e-6:
                best_cost = local_best_cost
                best_overlap = local_best_overlap
                best_state = dict(local_best_state)
                best_positions = self.reducer.evaluate(best_state)
                last_global_improve = time.time()

            # Bail early if we've got a good solution and little time left
            if time.time() - last_global_improve > stale_threshold and time_limit - (time.time() - start) < 20:
                break

        # Final post-process if the global best still has overlaps
        if best_positions is None and best_state is not None:
            best_positions = self.reducer.evaluate(best_state)

        return best_positions, best_cost


class Solver:
    """Top-level entry point. Mirrors the old API: Solver(data).solve(time_limit)
    returns (positions_dict, cost)."""

    def __init__(self, data):
        self.data = data
        self.boxes = [(i + 1, s[0], s[1]) for i, s in enumerate(data['box_size'])]
        self.n = len(self.boxes)
        self.w = {b[0]: b[1] for b in self.boxes}
        self.h = {b[0]: b[2] for b in self.boxes}
        self.nets = data.get('nets', [])

        self.sym_pairs_x = []
        self.sym_self_x = []
        for g in data.get('symmetry_x', []):
            for p in g.get('symmetry_pair', []):
                self.sym_pairs_x.append(tuple(p))
            for s in g.get('self_symmetry', []):
                self.sym_self_x.append(s)
        self.sym_pairs_y = []
        self.sym_self_y = []
        for g in data.get('symmetry_y', []):
            for p in g.get('symmetry_pair', []):
                self.sym_pairs_y.append(tuple(p))
            for s in g.get('self_symmetry', []):
                self.sym_self_y.append(s)

        al = data.get('align', {})
        self.align_left = al.get('left', [])
        self.align_right = al.get('right', [])
        self.align_top = al.get('top', [])
        self.align_bottom = al.get('bottom', [])

        self.repeat_groups = [rg['groups'] for rg in data.get('repeat_groups', [])]

        # Build constraint reducer once
        self.reducer = ConstraintReducer(data)

    def compute_overlap(self, pos: Dict[int, Tuple[float, float]]) -> float:
        """Total pairwise overlap area (used by main.py for validation)."""
        ids = list(pos.keys())
        total = 0.0
        for k in range(len(ids)):
            for j in range(k + 1, len(ids)):
                a, b = ids[k], ids[j]
                ox = min(pos[a][0] + self.w[a], pos[b][0] + self.w[b]) - max(pos[a][0], pos[b][0])
                oy = min(pos[a][1] + self.h[a], pos[b][1] + self.h[b]) - max(pos[a][1], pos[b][1])
                if ox > 0 and oy > 0:
                    total += ox * oy
        return total

    def compute_cost(self, pos: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
        hpwl = 0.0
        for net in self.nets:
            cxs = [pos[i][0] + self.w[i] / 2 for i in net if i in pos]
            cys = [pos[i][1] + self.h[i] / 2 for i in net if i in pos]
            if cxs and cys:
                hpwl += (max(cxs) - min(cxs)) + (max(cys) - min(cys))
        min_x = min(pos[i][0] for i in pos)
        max_x = max(pos[i][0] + self.w[i] for i in pos)
        min_y = min(pos[i][1] for i in pos)
        max_y = max(pos[i][1] + self.h[i] for i in pos)
        area = (max_x - min_x) * (max_y - min_y)
        return 10.0 * hpwl + area, hpwl, area

    def solve(self, time_limit: float = 120.0) -> Tuple[Optional[Dict[int, Tuple[float, float]]], float]:
        opt = Optimizer(self.reducer)
        return opt.solve(time_limit)
