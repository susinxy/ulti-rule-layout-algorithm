#!/usr/bin/env python3

import os
import sys
import json

from solver import load, Solver, plot_input, plot_output


def main():
    inp_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.json"
    data = load(inp_path)
    solver = Solver(data)

    plot_input(solver, "input_boxes.png")

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
    main()
