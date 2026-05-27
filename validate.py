import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def validate(inp, out):
    box_size = inp["box_size"]
    n = len(box_size)
    pos = out["box_position"]
    assert len(pos) == n, f"Expected {n} positions, got {len(pos)}"

    def xl(i): return pos[i-1][0]
    def yl(i): return pos[i-1][1]
    def xc(i): return xl(i) + box_size[i-1][0] / 2
    def yc(i): return yl(i) + box_size[i-1][1] / 2
    def xr(i): return xl(i) + box_size[i-1][0]
    def yr(i): return yl(i) + box_size[i-1][1]
    def w(i): return box_size[i-1][0]
    def h(i): return box_size[i-1][1]

    # Area
    min_x = min(xl(i) for i in range(1, n+1))
    max_x = max(xr(i) for i in range(1, n+1))
    min_y = min(yl(i) for i in range(1, n+1))
    max_y = max(yr(i) for i in range(1, n+1))
    area = (max_x - min_x) * (max_y - min_y)

    # HPWL
    hpwl = 0.0
    for net in inp["nets"]:
        cxs = [xc(i) for i in net]
        cys = [yc(i) for i in net]
        hpwl += (max(cxs) - min(cxs)) + (max(cys) - min(cys))

    cost = 10 * hpwl + area

    # Check overlap
    overlaps = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            EPS = 1e-4
            if (xr(i) - xl(j) > EPS and xr(j) - xl(i) > EPS and
                yr(i) - yl(j) > EPS and yr(j) - yl(i) > EPS):
                overlaps.append((i, j))

    # Symmetry x
    sym_errors = []
    axis_x_val = None
    for grp in inp.get("symmetry_x", []):
        pairs = grp.get("symmetry_pair", [])
        selfs = grp.get("self_symmetry", [])
        for a, b in pairs:
            mid = (xc(a) + xc(b)) / 2
            if axis_x_val is None:
                axis_x_val = mid
            if abs(mid - axis_x_val) > 0.01:
                sym_errors.append(f"sx pair ({a},{b}): mid_x={mid:.4f}, axis={axis_x_val:.4f}")
        for s in selfs:
            if axis_x_val is None:
                axis_x_val = xc(s)
            if abs(xc(s) - axis_x_val) > 0.01:
                sym_errors.append(f"sx self {s}: x_c={xc(s):.4f}, axis={axis_x_val:.4f}")

    # Symmetry y
    axis_y_val = None
    for grp in inp.get("symmetry_y", []):
        pairs = grp.get("symmetry_pair", [])
        selfs = grp.get("self_symmetry", [])
        for a, b in pairs:
            mid = (yc(a) + yc(b)) / 2
            if axis_y_val is None:
                axis_y_val = mid
            if abs(mid - axis_y_val) > 0.01:
                sym_errors.append(f"sy pair ({a},{b}): mid_y={mid:.4f}, axis={axis_y_val:.4f}")
        for s in selfs:
            if axis_y_val is None:
                axis_y_val = yc(s)
            if abs(yc(s) - axis_y_val) > 0.01:
                sym_errors.append(f"sy self {s}: y_c={yc(s):.4f}, axis={axis_y_val:.4f}")

    # Align
    align_errors = []
    al = inp.get("align", {})
    for grp in al.get("left", []):
        vals = [xl(i) for i in grp]
        if max(vals) - min(vals) > 0.01:
            align_errors.append(f"left {grp}: {vals}")
    for grp in al.get("right", []):
        vals = [xr(i) for i in grp]
        if max(vals) - min(vals) > 0.01:
            align_errors.append(f"right {grp}: {vals}")
    for grp in al.get("top", []):
        vals = [yr(i) for i in grp]
        if max(vals) - min(vals) > 0.01:
            align_errors.append(f"top {grp}: {vals}")
    for grp in al.get("bottom", []):
        vals = [yl(i) for i in grp]
        if max(vals) - min(vals) > 0.01:
            align_errors.append(f"bottom {grp}: {vals}")

    # Repeat groups
    rg_errors = []
    for rg in inp.get("repeat_groups", []):
        groups = rg["groups"]
        ref = groups[0]
        for gi, g in enumerate(groups[1:], 1):
            dx_vals = []
            dy_vals = []
            for k in range(len(ref)):
                a, b = ref[k], g[k]
                dx_vals.append(xl(a) - xl(b))
                dy_vals.append(yl(a) - yl(b))
                dx_vals.append(xc(a) - xc(b))
                dy_vals.append(yc(a) - yc(b))
                dx_vals.append(xr(a) - xr(b))
                dy_vals.append(yr(a) - yr(b))
            if max(dx_vals) - min(dx_vals) > 0.01:
                rg_errors.append(f"rg {gi}: dx spread {max(dx_vals)-min(dx_vals):.4f}")
            if max(dy_vals) - min(dy_vals) > 0.01:
                rg_errors.append(f"rg {gi}: dy spread {max(dy_vals)-min(dy_vals):.4f}")

    print(f"=== Validation Results ===")
    print(f"N boxes: {n}")
    print(f"Area: {area:.4f}")
    print(f"HPWL: {hpwl:.4f}")
    print(f"Cost (10*HPWL+Area): {cost:.4f}")
    print(f"Bounding box: x=[{min_x:.4f}, {max_x:.4f}], y=[{min_y:.4f}, {max_y:.4f}]")
    print(f"Symmetry axis x: {axis_x_val:.4f}")
    if axis_y_val is not None:
        print(f"Symmetry axis y: {axis_y_val:.4f}")
    print(f"Overlaps: {len(overlaps)} {overlaps[:5]}")
    print(f"Sym errors: {sym_errors}")
    print(f"Align errors: {align_errors}")
    print(f"Repeat group errors: {rg_errors}")


if __name__ == "__main__":
    inp_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "sample_output.json"
    inp = load(inp_path)
    out = load(out_path)
    validate(inp, out)
