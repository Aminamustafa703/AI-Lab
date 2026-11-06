
import heapq

graph = {"S": [("A", 1), ("B", 4)], "A": [("C", 2)],
         "B": [("G", 5)], "C": [("G", 3)], "G": []}
h = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}
true_remaining = {"S": 6, "A": 5, "B": 5, "C": 3, "G": 0}


def heuristic_audit():
    print("=== Admissibility: h(n) <= true remaining cost ===")
    for n in graph:
        ok = h[n] <= true_remaining[n]
        print(f"{n}: h={h[n]} true={true_remaining[n]} -> {'OK' if ok else 'FAIL'}")
    print("\n=== Consistency: h(u) <= cost(u,v) + h(v) ===")
    for u, edges in graph.items():
        for v, w in edges:
            ok = h[u] <= w + h[v]
            print(f"{u}->{v}: {h[u]} <= {w} + {h[v]} = {w + h[v]} -> {'OK' if ok else 'FAIL'}")


def astar_trace(start, goal, steps=3):
    """Prints g, h, f of each processed node and the frontier after expansion.
    Queue entries are (f, g, node, path); ties are broken by smaller g, then name."""
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    count = 0
    print(f"\n=== A* trace (first {steps} expansions) ===")
    print(f"Initial frontier: {start}(g=0, h={h[start]}, f={h[start]})")
    while pq and count < steps:
        f, g, node, path = heapq.heappop(pq)
        if g != best[node]:
            continue                                 
        count += 1
        print(f"\nExpansion {count}: select {node}: g={g}, h={h[node]}, f={g}+{h[node]}={f}")
        if node == goal:
            print("Goal selected - stop")
            return
        for nxt, w in graph[node]:
            ng = g + w
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(pq, (ng + h[nxt], ng, nxt, path + [nxt]))
                print(f"  generate {nxt}: g={g}+{w}={ng}, h={h[nxt]}, f={ng}+{h[nxt]}={ng + h[nxt]}")
        frontier = sorted(pq)
        print("  frontier:", ", ".join(f"{n}(g={gg},h={h[n]},f={ff})" for ff, gg, n, _ in frontier))
        print(f"  next selected: {frontier[0][2]} (smallest f = {frontier[0][0]})")


if __name__ == "__main__":
    heuristic_audit()
    astar_trace("S", "G", steps=3)