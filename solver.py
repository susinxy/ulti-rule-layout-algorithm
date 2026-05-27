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
        all_w = sum(self.w[b] for b in ids)

        # Global best across all runs
        global_best_pos = None
        global_best_cost = float('inf')
        global_best_alpha = None
        global_best_beta = None
        global_best_axis_x = None
        global_best_rg_offsets = None

        # Adaptive restart: trigger if no improvement in stale_seconds
        stale_seconds = 15.0
        last_improve_time = start
        run_count = 0

        while time.time() - start < time_limit - 2:
            run_count += 1
            is_first_run = run_count == 1

            # Initialization
            if is_first_run:
                indep_order = self.net_aware_order()
                alpha = list(indep_order) + [b for b in ids if b not in indep_order]
                beta = list(alpha)
                random.shuffle(beta)
                axis_x = all_w / 2.0 * 0.6
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
                # Warm restart: perturb global best, lower T
                alpha = list(global_best_alpha)
                beta = list(global_best_beta)
                axis_x = global_best_axis_x
                rg_offsets = dict(global_best_rg_offsets)

                # Perturb: swap random fraction of positions
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

            axis_y = None
            pos = self.decode_seq_pair(alpha, beta)
            self.apply_constraints(pos, axis_x, axis_y, rg_offsets)

            cost, hpwl, area = self.compute_cost(pos)
            overlap = self.compute_overlap(pos)
            score = cost + overlap * 5000

            local_best_pos = dict(pos)
            local_best_cost = cost if overlap < EPS else float('inf')
            local_best_alpha = list(alpha)
            local_best_beta = list(beta)
            local_best_axis_x = axis_x
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
                        local_best_rg_offsets = dict(rg_offsets)
                        chunk_last_improve = time.time()

                moves += 1
                T = max(T * cooling, 0.1)

                # Adaptive: break first run if stale and time permits restart
                if is_first_run and moves % 5000 == 0:
                    stale = time.time() - chunk_last_improve > stale_seconds
                    if stale and time_limit - (time.time() - start) > 25:
                        break

            # Update global best
            if local_best_cost < global_best_cost:
                global_best_cost = local_best_cost
                global_best_pos = dict(local_best_pos)
                global_best_alpha = list(local_best_alpha)
                global_best_beta = list(local_best_beta)
                global_best_axis_x = local_best_axis_x
                global_best_rg_offsets = dict(local_best_rg_offsets)
                last_improve_time = time.time()

            elapsed = time.time() - start
            print(f"Run {run_count}: local={local_best_cost:.0f} global={global_best_cost:.0f} {moves}it {elapsed:.0f}s", file=sys.stderr)

            remaining = time_limit - elapsed
            if remaining < 5:
                break

        return global_best_pos, global_best_cost


def plot_input(solver, filename="input_boxes.png"):
    """Visualize input box sizes in grid layout"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available, skipping plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    cols = int((solver.n ** 0.5) * 1.5)
    max_w = max(solver.w.values())
    max_h = max(solver.h.values())
    cell_w = max_w * 1.3
    cell_h = max_h * 1.3

    colors = plt.cm.Set3([(i * 0.1) % 1 for i in range(solver.n)])

    for i, bid in enumerate(range(1, solver.n + 1)):
        row = i // cols
        col = i % cols
        x = col * cell_w
        y = row * cell_h
        w = solver.w[bid]
        h = solver.h[bid]

        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black',
                                  facecolor=colors[i % len(colors)], alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, str(bid), ha='center', va='center',
                fontsize=8, weight='bold')

    ax.set_title(f"Input Boxes (n={solver.n})")
    ax.set_aspect('equal')
    ax.autoscale()
    ax.margins(0.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Input plot saved: {filename}", file=sys.stderr)


def plot_output(solver, pos, filename="output_layout.png"):
    """Visualize output layout with symmetry axes and nets"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available, skipping plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    colors = plt.cm.Set3([(i * 0.07) % 1 for i in range(solver.n)])

    for i, bid in enumerate(range(1, solver.n + 1)):
        if bid not in pos:
            continue
        x, y = pos[bid]
        w = solver.w[bid]
        h = solver.h[bid]

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black',
                                  facecolor=colors[i % len(colors)])
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, str(bid), ha='center', va='center',
                fontsize=7, weight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                                                      facecolor='white', edgecolor='none', alpha=0.8))

    # Draw symmetry axes
    if solver.sym_pairs_x or solver.sym_self_x:
        xs = [pos[bid][0] + solver.w[bid] / 2 for bid in pos]
        axis_x = sum(xs) / len(xs)
        ymin = min(pos[bid][1] for bid in pos) - 5
        ymax = max(pos[bid][1] + solver.h[bid] for bid in pos) + 5
        ax.axvline(axis_x, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='X-axis sym')

    if solver.sym_pairs_y or solver.sym_self_y:
        ys = [pos[bid][1] + solver.h[bid] / 2 for bid in pos]
        axis_y = sum(ys) / len(ys)
        xmin = min(pos[bid][0] for bid in pos) - 5
        xmax = max(pos[bid][0] + solver.w[bid] for bid in pos) + 5
        ax.axhline(axis_y, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Y-axis sym')

    # Draw nets as faint connections
    net_colors = plt.cm.tab20([i * 0.05 % 1 for i in range(len(solver.nets))])
    for ni, net in enumerate(solver.nets):
        centers = [(pos[bid][0] + solver.w[bid] / 2, pos[bid][1] + solver.h[bid] / 2)
                   for bid in net if bid in pos]
        if len(centers) >= 2:
            cx = [c[0] for c in centers]
            cy = [c[1] for c in centers]
            ax.plot(cx, cy, '-', color=net_colors[ni], alpha=0.3, linewidth=1)

    ax.set_title(f"Output Layout (Cost: {solver.compute_cost(pos)[0]:.0f})")
    ax.set_aspect('equal')
    ax.autoscale()
    ax.margins(0.05)
    if solver.sym_pairs_x or solver.sym_pairs_y:
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Output plot saved: {filename}", file=sys.stderr)


def main():
    inp_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    data = load(inp_path)
    solver = Solver(data)

    plot_input(solver, "input_boxes.png")

    # Plot sample output if available
    sample_output_path = "sample_output.json"
    if os.path.exists(sample_output_path):
        try:
            sample_data = load(sample_output_path)
            sample_pos = {i+1: tuple(p) for i, p in enumerate(sample_data["box_position"])}
            plot_output(solver, sample_pos, "sample_output_layout.png")
            print(f"Sample output plot saved: sample_output_layout.png", file=sys.stderr)
        except Exception as e:
            print(f"Failed to plot sample output: {e}", file=sys.stderr)

    pos, cost = solver.solve(120)

    result = {"box_position": [[round(pos[i][0], 4), round(pos[i][1], 4)] for i in range(1, solver.n + 1)]}
    print(json.dumps(result, indent=2))

    final_cost, hpwl, area = solver.compute_cost(pos)
    overlap = solver.compute_overlap(pos)
    print(f"Cost: {final_cost:.2f} HPWL: {hpwl:.2f} Area: {area:.2f} Ovl: {overlap:.6f}", file=sys.stderr)

    plot_output(solver, pos, "output_layout.png")


if __name__ == "__main__":
    import os
    main()
