import math, random, time, sys
from collections import defaultdict

EPS = 1e-6

class Solver:
    def __init__(self, data):
        self.boxes = [(i+1, s[0], s[1]) for i, s in enumerate(data["box_size"])]
        self.n = len(self.boxes)
        self.w = {b[0]: b[1] for b in self.boxes}
        self.h = {b[0]: b[2] for b in self.boxes}
        self.nets = data.get("nets", [])

        self.sym_pairs_x = []
        self.sym_self_x = []
        for g in data.get("symmetry_x", []):
            for p in g.get("symmetry_pair", []): self.sym_pairs_x.append(tuple(p))
            for s in g.get("self_symmetry", []): self.sym_self_x.append(s)
        self.sym_pairs_y = []
        self.sym_self_y = []
        for g in data.get("symmetry_y", []):
            for p in g.get("symmetry_pair", []): self.sym_pairs_y.append(tuple(p))
            for s in g.get("self_symmetry", []): self.sym_self_y.append(s)

        al = data.get("align", {})
        self.align_left = al.get("left", [])
        self.align_right = al.get("right", [])
        self.align_top = al.get("top", [])
        self.align_bottom = al.get("bottom", [])

        self.repeat_groups = [rg["groups"] for rg in data.get("repeat_groups", [])]

        # dependent boxes: derived from symmetry / repeat-group (not optimized directly)
        self.dependent = set()
        for a, b in self.sym_pairs_x: self.dependent.add(b)
        for a, b in self.sym_pairs_y: self.dependent.add(b)
        for rg in self.repeat_groups:
            for slave in rg[1:]:
                for s in slave: self.dependent.add(s)

        self.indep = [b[0] for b in self.boxes if b[0] not in self.dependent]

        # net connectivity weight (used for net-aware initial ordering)
        self.net_neighbors = defaultdict(lambda: defaultdict(int))
        for net in self.nets:
            for i in net:
                for j in net:
                    if i != j: self.net_neighbors[i][j] += 1

        # master-of maps (informational, not currently used to derive slaves)
        self.sym_pair_master = {b: a for a, b in self.sym_pairs_x}
        self.sym_pair_master.update({b: a for a, b in self.sym_pairs_y})

        self.rg_master_of = {}
        for rg in self.repeat_groups:
            master = rg[0]
            for slave in rg[1:]:
                for a, b in zip(master, slave):
                    self.rg_master_of[b] = (master, slave)

        # Forced offsets: if master and slave in same repeat group share an alignment
        # constraint and have the same size, the offset is forced (e.g. dx=0 for
        # right-align with equal widths).
        self.forced_offsets = {}
        for rg in self.repeat_groups:
            master = rg[0]
            for gi, slave in enumerate(rg[1:]):
                key = (master[0], slave[0], gi)
                forced_dx = None
                forced_dy = None

                for grp in self.align_right:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp and abs(self.w[ma] - self.w[sl]) < EPS:
                            forced_dx = 0.0

                for grp in self.align_left:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            forced_dx = 0.0

                for grp in self.align_top:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp and abs(self.h[ma] - self.h[sl]) < EPS:
                            forced_dy = 0.0

                for grp in self.align_bottom:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            forced_dy = 0.0

                if forced_dx is not None or forced_dy is not None:
                    self.forced_offsets[key] = (forced_dx, forced_dy)

    def decode_seq_pair(self, alpha, beta):
        """Decode sequence pair (alpha, beta) to positions using longest path."""
        n = len(alpha)
        beta_pos = {x: i for i, x in enumerate(beta)}

        x = {item: 0.0 for item in alpha}
        for i in range(n):
            ai = alpha[i]
            for j in range(i + 1, n):
                aj = alpha[j]
                if beta_pos[ai] < beta_pos[aj]:
                    val = x[ai] + self.w[ai]
                    if val > x[aj]:
                        x[aj] = val

        y = {item: 0.0 for item in alpha}
        for i in range(n):
            ai = alpha[i]
            for j in range(i + 1, n):
                aj = alpha[j]
                if beta_pos[ai] > beta_pos[aj]:
                    val = y[ai] + self.h[ai]
                    if val > y[aj]:
                        y[aj] = val

        return {alpha[i]: (x[alpha[i]], y[alpha[i]]) for i in range(n)}

    def apply_constraints(self, pos, axis_x, axis_y, rg_offsets):
        """Apply hard constraints with correct ordering to avoid conflicts.
        
        Priority order (highest to lowest):
        1. Symmetry (must enforce axis, cannot be overridden)
        2. Repeat groups (slaves follow masters)
        3. Alignment (lowest priority, can be overridden by symmetry)
        
        Strategy:
        - Apply alignment first (can be overridden)
        - Apply repeat groups to derive slaves
        - Apply symmetry to enforce axis (highest priority, overrides all)
        - Re-apply repeat groups so slaves follow corrected masters
        - Final symmetry pass to ensure no drift
        """

        slave_boxes = set()
        for rg in self.repeat_groups:
            for slave in rg[1:]:
                slave_boxes.update(slave)

        # Helper: apply alignment constraints
        def _apply_alignment():
            for grp in self.align_left:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    t = min(pos[i][0] for i in non_slave)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (t, pos[i][1])
            for grp in self.align_right:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    t = max(pos[i][0] + self.w[i] for i in non_slave)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (t - self.w[i], pos[i][1])
            for grp in self.align_top:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    t = max(pos[i][1] + self.h[i] for i in non_slave)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (pos[i][0], t - self.h[i])
            for grp in self.align_bottom:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    t = min(pos[i][1] for i in non_slave)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (pos[i][0], t)

        # Helper: apply repeat groups
        def _apply_repeat_groups():
            for rg in self.repeat_groups:
                master = rg[0]
                for gi, slave in enumerate(rg[1:]):
                    key = (master[0], slave[0], gi)
                    forced = self.forced_offsets.get(key, (None, None))

                    if key not in rg_offsets:
                        if master[0] in pos and slave[0] in pos:
                            dx = pos[slave[0]][0] - pos[master[0]][0]
                            dy = pos[slave[0]][1] - pos[master[0]][1]
                            if forced[0] is not None: dx = forced[0]
                            if forced[1] is not None: dy = forced[1]
                            rg_offsets[key] = (dx, dy)
                        else:
                            dx = forced[0] if forced[0] is not None else 100.0
                            dy = forced[1] if forced[1] is not None else 50.0
                            rg_offsets[key] = (dx, dy)

                    dx, dy = rg_offsets[key]
                    if forced[0] is not None: dx = forced[0]
                    if forced[1] is not None: dy = forced[1]
                    rg_offsets[key] = (dx, dy)

                    for ma, sl in zip(master, slave):
                        if ma in pos:
                            pos[sl] = (pos[ma][0] + dx, pos[ma][1] + dy)

        # Helper: apply symmetry (highest priority)
        def _apply_symmetry():
            # X-axis symmetry
            for a, b in self.sym_pairs_x:
                if a in pos:
                    xca = pos[a][0] + self.w[a] / 2.0
                    yca = pos[a][1] + self.h[a] / 2.0
                    xcb = 2.0 * axis_x - xca
                    pos[b] = (xcb - self.w[b] / 2.0, yca - self.h[b] / 2.0)
            for s in self.sym_self_x:
                if s in pos:
                    pos[s] = (axis_x - self.w[s] / 2.0, pos[s][1])

            # Y-axis symmetry
            if axis_y is not None:
                for a, b in self.sym_pairs_y:
                    if a in pos:
                        yca = pos[a][1] + self.h[a] / 2.0
                        xca = pos[a][0] + self.w[a] / 2.0
                        ycb = 2.0 * axis_y - yca
                        pos[b] = (xca - self.w[b] / 2.0, ycb - self.h[b] / 2.0)
                for s in self.sym_self_y:
                    if s in pos:
                        pos[s] = (pos[s][0], axis_y - self.h[s] / 2.0)

        # Iterative application to converge
        for _iteration in range(5):
            # Phase 1: Alignment (lowest priority, can be overridden)
            _apply_alignment()

            # Phase 2: Repeat groups (first pass)
            _apply_repeat_groups()

            # Phase 3: Symmetry (highest priority, enforces axis)
            _apply_symmetry()

            # Phase 4: Repeat groups (second pass, slaves follow corrected masters)
            _apply_repeat_groups()

            # Phase 5: Final symmetry (ensure no drift from repeat group changes)
            _apply_symmetry()

    def compute_cost(self, pos):
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
        return 10 * hpwl + area, hpwl, area

    def compute_overlap(self, pos):
        ids = list(pos.keys())
        total = 0.0
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                ox = min(pos[a][0]+self.w[a], pos[b][0]+self.w[b]) - max(pos[a][0], pos[b][0])
                oy = min(pos[a][1]+self.h[a], pos[b][1]+self.h[b]) - max(pos[a][1], pos[b][1])
                if ox > EPS and oy > EPS:
                    total += ox * oy
        return total

    def net_aware_order(self):
        """BFS order by net connectivity for good initial alpha."""
        neighbors = self.net_neighbors
        start = max(self.indep, key=lambda x: sum(neighbors[x].values()))
        visited = set()
        order = []
        queue = [start]
        while queue:
            curr = queue.pop(0)
            if curr in visited: continue
            visited.add(curr)
            order.append(curr)
            nbs = sorted(neighbors[curr].keys(), key=lambda n: -neighbors[curr][n])
            for nb in nbs:
                if nb not in visited and nb in self.indep:
                    queue.append(nb)
        for i in self.indep:
            if i not in visited: order.append(i)
        return order

    def solve(self, time_limit=120):
        """Run adaptive-restart SA. Returns (positions_dict, best_cost)."""
        start = time.time()
        ids = list(range(1, self.n + 1))
        n_ids = len(ids)
        all_w = sum(self.w[b] for b in ids)
        all_h = sum(self.h[b] for b in ids)
        has_sym_y = bool(self.sym_pairs_y or self.sym_self_y)

        global_best_pos = None
        global_best_cost = float('inf')
        global_best_alpha = None
        global_best_beta = None
        global_best_axis_x = None
        global_best_axis_y = None
        global_best_rg_offsets = None

        stale_seconds = 15.0
        last_improve_time = start
        run_count = 0

        while time.time() - start < time_limit - 2:
            run_count += 1
            is_first_run = run_count == 1

            if is_first_run:
                indep_order = self.net_aware_order()
                alpha = list(indep_order) + [b for b in ids if b not in indep_order]
                beta = list(alpha)
                random.shuffle(beta)
                axis_x = all_w / 2.0 * 0.6
                axis_y = None
                rg_offsets = {}
                for rg in self.repeat_groups:
                    master = rg[0]
                    for gi, slave in enumerate(rg[1:]):
                        key = (master[0], slave[0], gi)
                        forced = self.forced_offsets.get(key, (None, None))
                        total_w = sum(self.w[m] for m in master)
                        dx = forced[0] if forced[0] is not None else total_w + 20
                        dy = forced[1] if forced[1] is not None else 0.0
                        rg_offsets[key] = (dx, dy)
                T_start = 2000.0
                cooling = 0.99997
            else:
                alpha = list(global_best_alpha)
                beta = list(global_best_beta)
                axis_x = global_best_axis_x
                rg_offsets = dict(global_best_rg_offsets)

                perturb_frac = random.uniform(0.05, 0.15)
                n_swap = max(1, int(len(ids) * perturb_frac))
                for _ in range(n_swap):
                    i, j = random.sample(range(n_ids), 2)
                    alpha[i], alpha[j] = alpha[j], alpha[i]
                for _ in range(n_swap):
                    i, j = random.sample(range(n_ids), 2)
                    beta[i], beta[j] = beta[j], beta[i]
                axis_x += random.gauss(0, 5.0)

                T_start = random.uniform(300, 600)
                cooling = 0.999998 if time.time() - start > 60 else 0.99999

            pos = self.decode_seq_pair(alpha, beta)
            axis_y = all_h / 2.0 * 0.6 if (has_sym_y and is_first_run) else axis_y
            if not has_sym_y:
                axis_y = None
            elif not is_first_run:
                axis_y = global_best_axis_y + random.gauss(0, 5.0) if global_best_axis_y is not None else all_h / 2.0 * 0.6
            self.apply_constraints(pos, axis_x, axis_y, rg_offsets)

            cost, hpwl, area = self.compute_cost(pos)
            overlap = self.compute_overlap(pos)
            score = cost + overlap * 5000

            local_best_pos = dict(pos)
            local_best_cost = cost if overlap < EPS else float('inf')
            local_best_alpha = list(alpha)
            local_best_beta = list(beta)
            local_best_axis_x = axis_x
            local_best_axis_y = axis_y if has_sym_y else None
            local_best_rg_offsets = dict(rg_offsets)

            T = T_start
            moves = 0
            chunk_start = time.time()
            chunk_last_improve = chunk_start
            chunk_budget = time_limit - (chunk_start - start) - 1 if is_first_run else min(30.0, time_limit - (chunk_start - start) - 1)

            while time.time() - chunk_start < chunk_budget and time.time() - start < time_limit - 1:
                new_alpha = list(alpha)
                new_beta = list(beta)
                new_axis_x = axis_x
                new_axis_y = axis_y
                new_rg_offsets = dict(rg_offsets)

                r = random.random()
                if r < 0.25:
                    i, j = random.sample(range(n_ids), 2)
                    new_alpha[i], new_alpha[j] = new_alpha[j], new_alpha[i]
                elif r < 0.50:
                    i, j = random.sample(range(n_ids), 2)
                    new_beta[i], new_beta[j] = new_beta[j], new_beta[i]
                elif r < 0.60:
                    if len(self.indep) >= 2:
                        a_pos = random.sample(range(n_ids), 2)
                        new_alpha[a_pos[0]], new_alpha[a_pos[1]] = new_alpha[a_pos[1]], new_alpha[a_pos[0]]
                elif r < 0.70:
                    step = max(5.0, T / 100.0)
                    if has_sym_y and random.random() < 0.5:
                        new_axis_y = (new_axis_y or 0.0) + random.gauss(0, step)
                    else:
                        new_axis_x += random.gauss(0, step)
                elif r < 0.80:
                    i, j = sorted(random.sample(range(n_ids), 2))
                    if random.random() < 0.5:
                        new_alpha[i:j+1] = reversed(new_alpha[i:j+1])
                    else:
                        new_beta[i:j+1] = reversed(new_beta[i:j+1])
                elif r < 0.90:
                    i = random.randint(0, n_ids - 1)
                    j = random.randint(0, n_ids - 1)
                    val = new_alpha.pop(i)
                    new_alpha.insert(j, val)
                else:
                    step = max(10.0, T / 50.0)
                    for key in list(new_rg_offsets.keys()):
                        if random.random() < 0.5:
                            ox, oy = new_rg_offsets[key]
                            forced = self.forced_offsets.get(key, (None, None))
                            new_ox = forced[0] if forced[0] is not None else ox + random.gauss(0, step)
                            new_oy = forced[1] if forced[1] is not None else oy + random.gauss(0, step)
                            new_rg_offsets[key] = (new_ox, new_oy)

                new_pos = self.decode_seq_pair(new_alpha, new_beta)
                self.apply_constraints(new_pos, new_axis_x, new_axis_y, new_rg_offsets)

                new_cost, _, _ = self.compute_cost(new_pos)
                new_overlap = self.compute_overlap(new_pos)
                new_score = new_cost + new_overlap * 5000

                delta = new_score - score
                if delta < 0 or random.random() < math.exp(-delta / max(T, 0.1)):
                    alpha = new_alpha
                    beta = new_beta
                    axis_x = new_axis_x
                    axis_y = new_axis_y
                    rg_offsets = new_rg_offsets
                    pos = new_pos
                    cost = new_cost
                    overlap = new_overlap
                    score = new_score

                    if overlap < EPS and cost < local_best_cost:
                        local_best_cost = cost
                        local_best_pos = dict(pos)
                        local_best_alpha = list(alpha)
                        local_best_beta = list(beta)
                        local_best_axis_x = axis_x
                        local_best_axis_y = axis_y if has_sym_y else None
                        local_best_rg_offsets = dict(rg_offsets)
                        chunk_last_improve = time.time()

                moves += 1
                T = max(T * cooling, 0.1)

                if is_first_run and moves % 5000 == 0:
                    if time.time() - chunk_last_improve > stale_seconds and time_limit - (time.time() - start) > 25:
                        break

            if local_best_cost < global_best_cost:
                global_best_cost = local_best_cost
                global_best_pos = dict(local_best_pos)
                global_best_alpha = list(local_best_alpha)
                global_best_beta = list(local_best_beta)
                global_best_axis_x = local_best_axis_x
                global_best_axis_y = local_best_axis_y if has_sym_y else None
                global_best_rg_offsets = dict(local_best_rg_offsets)
                last_improve_time = time.time()

            elapsed = time.time() - start
            print(f"Run {run_count}: local={local_best_cost:.0f} global={global_best_cost:.0f} {moves}it {elapsed:.0f}s", file=sys.stderr)

            if time_limit - elapsed < 5:
                break

        return global_best_pos, global_best_cost
