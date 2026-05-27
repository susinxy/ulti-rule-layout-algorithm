import json, math, random, time, sys
from collections import defaultdict

EPS = 1e-6

def load(p):
    with open(p) as f: return json.load(f)

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

        self.dependent = set()
        for a, b in self.sym_pairs_x: self.dependent.add(b)
        for a, b in self.sym_pairs_y: self.dependent.add(b)
        for rg in self.repeat_groups:
            for slave in rg[1:]:
                for s in slave: self.dependent.add(s)

        self.indep = [b[0] for b in self.boxes if b[0] not in self.dependent]

        # Net connectivity weight
        self.net_neighbors = defaultdict(lambda: defaultdict(int))
        for net in self.nets:
            for i in net:
                for j in net:
                    if i != j: self.net_neighbors[i][j] += 1

        # Map boxes to constraints
        self.sym_pair_master = {}
        for a, b in self.sym_pairs_x: self.sym_pair_master[b] = a
        for a, b in self.sym_pairs_y: self.sym_pair_master[b] = a

        self.rg_master_of = {}
        for rg in self.repeat_groups:
            master = rg[0]
            for slave in rg[1:]:
                for a, b in zip(master, slave):
                    self.rg_master_of[b] = (master, slave)

        # Analyze forced offsets from alignment constraints
        # If master and slave in same group are aligned and have same size, offset is forced
        self.forced_offsets = {}  # key -> (dx, dy) with None for free dimensions
        for rg in self.repeat_groups:
            master = rg[0]
            for gi, slave in enumerate(rg[1:]):
                key = (master[0], slave[0], gi)
                forced_dx = None
                forced_dy = None

                # Check right alignment
                for grp in self.align_right:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            # ma and sl must have same right edge
                            # x_ma + w_ma = x_sl + w_sl = (x_ma + dx) + w_sl
                            # If w_ma = w_sl, then dx = 0
                            if abs(self.w[ma] - self.w[sl]) < EPS:
                                forced_dx = 0.0

                # Check left alignment
                for grp in self.align_left:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            # x_ma = x_sl = x_ma + dx => dx = 0
                            forced_dx = 0.0

                # Check top alignment
                for grp in self.align_top:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            # y_ma + h_ma = y_sl + h_sl = (y_ma + dy) + h_sl
                            # If h_ma = h_sl, then dy = 0
                            if abs(self.h[ma] - self.h[sl]) < EPS:
                                forced_dy = 0.0

                # Check bottom alignment
                for grp in self.align_bottom:
                    for ma, sl in zip(master, slave):
                        if ma in grp and sl in grp:
                            # y_ma = y_sl => dy = 0
                            forced_dy = 0.0

                if forced_dx is not None or forced_dy is not None:
                    self.forced_offsets[key] = (forced_dx, forced_dy)

    def decode_seq_pair(self, alpha, beta):
        """Decode sequence pair (alpha, beta) to positions using longest path."""
        n = len(alpha)
        alpha_pos = {x: i for i, x in enumerate(alpha)}
        beta_pos = {x: i for i, x in enumerate(beta)}

        # X: i before j in alpha AND i before j in beta => x[j] >= x[i] + w[i]
        # Y: i before j in alpha AND j before i in beta => y[j] >= y[i] + h[i]
        x = {}
        for item in alpha:
            x[item] = 0.0

        # Topological pass for X
        for i in range(n):
            ai = alpha[i]
            for j in range(i + 1, n):
                aj = alpha[j]
                if beta_pos[ai] < beta_pos[aj]:
                    val = x[ai] + self.w[ai]
                    if val > x[aj]:
                        x[aj] = val

        y = {}
        for item in alpha:
            y[item] = 0.0

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
        """Apply hard constraints with correct ordering to avoid conflicts."""
        
        # Build set of all slave boxes (should not be modified by alignment)
        slave_boxes = set()
        for rg in self.repeat_groups:
            for slave in rg[1:]:
                slave_boxes.update(slave)
        
        for iteration in range(10):
            # PHASE 1: Symmetry on masters and self-symmetric boxes
            for a, b in self.sym_pairs_x:
                if a in pos and b not in slave_boxes:
                    xca = pos[a][0] + self.w[a] / 2.0
                    yca = pos[a][1] + self.h[a] / 2.0
                    xcb = 2.0 * axis_x - xca
                    pos[b] = (xcb - self.w[b] / 2.0, yca - self.h[b] / 2.0)
            for s in self.sym_self_x:
                if s in pos and s not in slave_boxes:
                    pos[s] = (axis_x - self.w[s] / 2.0, pos[s][1])

            if axis_y is not None:
                for a, b in self.sym_pairs_y:
                    if a in pos and b not in slave_boxes:
                        yca = pos[a][1] + self.h[a] / 2.0
                        xca = pos[a][0] + self.w[a] / 2.0
                        ycb = 2.0 * axis_y - yca
                        pos[b] = (xca - self.w[b] / 2.0, ycb - self.h[b] / 2.0)
                for s in self.sym_self_y:
                    if s in pos and s not in slave_boxes:
                        pos[s] = (pos[s][0], axis_y - self.h[s] / 2.0)

            # PHASE 2: Alignment on non-slave boxes only
            for grp in self.align_left:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    vals = [pos[i][0] for i in non_slave]
                    t = min(vals)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (t, pos[i][1])
            for grp in self.align_right:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    vals = [pos[i][0] + self.w[i] for i in non_slave]
                    t = max(vals)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (t - self.w[i], pos[i][1])
            for grp in self.align_top:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    vals = [pos[i][1] + self.h[i] for i in non_slave]
                    t = max(vals)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (pos[i][0], t - self.h[i])
            for grp in self.align_bottom:
                non_slave = [i for i in grp if i in pos and i not in slave_boxes]
                if non_slave:
                    vals = [pos[i][1] for i in non_slave]
                    t = min(vals)
                    for i in grp:
                        if i in pos and i not in slave_boxes:
                            pos[i] = (pos[i][0], t)

            # PHASE 3: Repeat groups (derive ALL slaves from masters)
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

                    # Apply forced offsets
                    dx, dy = rg_offsets[key]
                    if forced[0] is not None: dx = forced[0]
                    if forced[1] is not None: dy = forced[1]
                    rg_offsets[key] = (dx, dy)

                    for ma, sl in zip(master, slave):
                        if ma in pos:
                            pos[sl] = (pos[ma][0] + dx, pos[ma][1] + dy)

            # PHASE 4: Apply symmetry to slave boxes that also have symmetry constraints
            for a, b in self.sym_pairs_x:
                if a in pos and b in slave_boxes:
                    xca = pos[a][0] + self.w[a] / 2.0
                    yca = pos[a][1] + self.h[a] / 2.0
                    xcb = 2.0 * axis_x - xca
                    pos[b] = (xcb - self.w[b] / 2.0, yca - self.h[b] / 2.0)

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
        """BFS order by net connectivity for good initial alpha"""
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
        start = time.time()
        ids = list(range(1, self.n + 1))
        n_ids = len(ids)

        indep_order = self.net_aware_order()
        alpha = list(indep_order) + [b for b in ids if b not in indep_order]
        beta = list(alpha)
        random.shuffle(beta)

        pos = self.decode_seq_pair(alpha, beta)

        all_w = sum(self.w[b] for b in ids)
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

        self.apply_constraints(pos, axis_x, axis_y, rg_offsets)

        cost, hpwl, area = self.compute_cost(pos)
        overlap = self.compute_overlap(pos)
        score = cost + overlap * 5000

        best_pos = dict(pos)
        best_cost = cost if overlap < EPS else float('inf')
        best_alpha = list(alpha)
        best_beta = list(beta)
        best_axis_x = axis_x
        best_rg_offsets = dict(rg_offsets)

        T = 2000.0
        cooling = 0.99997
        moves = 0

        while time.time() - start < time_limit:
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
                step = max(10.0, T / 100.0)
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
                step = max(15.0, T / 50.0)
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

                if overlap < EPS and cost < best_cost:
                    best_cost = cost
                    best_pos = dict(pos)
                    best_alpha = list(alpha)
                    best_beta = list(beta)
                    best_axis_x = axis_x
                    best_rg_offsets = dict(rg_offsets)

            moves += 1
            T = max(T * cooling, 0.1)

            if moves % 10000 == 0:
                elapsed = time.time() - start
                print(f"It {moves} T={T:.2f} Best={best_cost:.0f} Cur={cost:.0f} Ovl={overlap:.1f} {elapsed:.0f}s", file=sys.stderr)

        return best_pos, best_cost


def main():
    inp_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    data = load(inp_path)
    solver = Solver(data)
    pos, cost = solver.solve(120)

    result = {"box_position": [[round(pos[i][0], 4), round(pos[i][1], 4)] for i in range(1, solver.n + 1)]}
    print(json.dumps(result, indent=2))

    final_cost, hpwl, area = solver.compute_cost(pos)
    overlap = solver.compute_overlap(pos)
    print(f"Cost: {final_cost:.2f} HPWL: {hpwl:.2f} Area: {area:.2f} Ovl: {overlap:.6f}", file=sys.stderr)

if __name__ == "__main__":
    main()
