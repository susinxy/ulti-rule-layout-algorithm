import os
import sys
import json

from solver import Solver


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_input(solver, filename):
    """Visualize input box sizes in a grid layout."""
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


def plot_output(solver, pos, filename):
    """Visualize layout with symmetry axes and nets."""
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
                fontsize=7, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8))

    if solver.sym_pairs_x or solver.sym_self_x:
        xs = [pos[bid][0] + solver.w[bid] / 2 for bid in pos]
        axis_x = sum(xs) / len(xs)
        ax.axvline(axis_x, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='X-axis sym')

    if solver.sym_pairs_y or solver.sym_self_y:
        ys = [pos[bid][1] + solver.h[bid] / 2 for bid in pos]
        axis_y = sum(ys) / len(ys)
        ax.axhline(axis_y, color='green', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Y-axis sym')

    net_colors = plt.cm.tab20([i * 0.05 % 1 for i in range(len(solver.nets))])
    for ni, net in enumerate(solver.nets):
        centers = [(pos[bid][0] + solver.w[bid] / 2,
                    pos[bid][1] + solver.h[bid] / 2)
                   for bid in net if bid in pos]
        if len(centers) >= 2:
            cx = [c[0] for c in centers]
            cy = [c[1] for c in centers]
            ax.plot(cx, cy, '-', color=net_colors[ni], alpha=0.3, linewidth=1)

    cost = solver.compute_cost(pos)[0]
    ax.set_title(f"Output Layout (Cost: {cost:.0f})")
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
    # Parse argument: case name, directory path, or JSON file path
    arg = sys.argv[1] if len(sys.argv) > 1 else "sample"
    
    # Determine input path and case name
    if arg.endswith('.json') and os.path.isfile(arg):
        # Direct JSON file
        input_path = arg
        case_name = os.path.splitext(os.path.basename(arg))[0]
    elif os.path.isdir(arg):
        # Directory containing input.json
        case_dir = arg.rstrip('/')
        input_path = os.path.join(case_dir, 'input.json')
        case_name = os.path.basename(case_dir)
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found", file=sys.stderr)
            sys.exit(1)
    else:
        # Case name (look in cases/ directory)
        case_name = arg
        input_path = os.path.join('cases', case_name, 'input.json')
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found", file=sys.stderr)
            sys.exit(1)
    
    # Create results directory
    results_dir = os.path.join('results', case_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load input data
    data = load(input_path)
    solver = Solver(data)
    
    # Plot input
    input_plot_path = os.path.join(results_dir, 'input_boxes.png')
    plot_input(solver, input_plot_path)
    
    # Check for expected output and plot if exists
    expected_path = os.path.join('cases', case_name, 'expected.json')
    if os.path.exists(expected_path):
        try:
            expected_data = load(expected_path)
            expected_pos = {i + 1: tuple(p) for i, p in enumerate(expected_data["box_position"])}
            expected_plot_path = os.path.join(results_dir, 'expected_layout.png')
            plot_output(solver, expected_pos, expected_plot_path)
            print(f"Expected layout plot saved: {expected_plot_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to plot expected output: {e}", file=sys.stderr)
    
    # Solve
    pos, cost = solver.solve(120)
    
    # Format and output result
    result = {
        "box_position": [
            [round(pos[i][0], 4), round(pos[i][1], 4)]
            for i in range(1, solver.n + 1)
        ]
    }
    print(json.dumps(result, indent=2))
    
    # Save result to file
    output_path = os.path.join(results_dir, 'output.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print statistics
    final_cost, hpwl, area = solver.compute_cost(pos)
    overlap = solver.compute_overlap(pos)
    print(
        f"Cost: {final_cost:.2f} HPWL: {hpwl:.2f} Area: {area:.2f} Ovl: {overlap:.6f}",
        file=sys.stderr,
    )
    
    # Plot output
    output_plot_path = os.path.join(results_dir, 'output_layout.png')
    plot_output(solver, pos, output_plot_path)


if __name__ == "__main__":
    main()
