import heapq
from collections import deque
graph = {
    "A": [("B", 1), ("C", 5)],
    "B": [("D", 4), ("E", 1)],
    "C": [("F", 3), ("G", 7)],     # C -> G is the added corridor
    "D": [],
    "E": [("G", 1)],
    "F": [],
    "G": [],
}

def check_weights(g):
    for u, edges in g.items():
        for v, w in edges:
            if w < 0:
                raise ValueError(f"negative edge {u}->{v} ({w}): UCS needs nonnegative costs")


def ucs(g, start, goal):
    """Returns (path, cost, processing_order, trace)."""
    check_weights(g)
    counter = 0                                   # keeps tie-break order stable
    frontier = [(0, counter, start, [start])]     # (cost, order, node, path)
    best = {start: 0}
    done, order, trace = set(), [], []
    while frontier:
        cost, _, node, path = heapq.heappop(frontier)
        if node in done:                          # stale duplicate entry
            continue
        done.add(node)
        order.append(node)
        if node == goal:
            trace.append((node, cost, "goal reached"))
            return path, cost, order, trace
        for nxt, w in g[node]:
            new_cost = cost + w
            if nxt not in done and new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, nxt, path + [nxt]))
        snap = sorted((c, n) for c, _, n, _ in frontier if n not in done)
        trace.append((node, cost, ", ".join(f"{n}({c})" for c, n in snap)))
    return None, float("inf"), order, trace


def bfs(g, start, goal):
    """BFS ignores weights: fewest edges. Returns (path, order)."""
    q, seen, order = deque([(start, [start])]), {start}, []
    while q:
        node, path = q.popleft()
        order.append(node)
        if node == goal:
            return path, order
        for nxt, _ in g[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
    return None, order


def path_cost(g, path):
    """Independent check: add up edge weights along a path."""
    total = 0
    for u, v in zip(path, path[1:]):
        total += dict(g[u])[v]
    return total


if __name__ == "__main__":
    print("=== Weighted graph (edge: weight) ===")
    for u, edges in graph.items():
        for v, w in edges:
            print(f"{u}->{v}: {w}")

    path, cost, order, trace = ucs(graph, "A", "G")
    print("\n=== UCS trace (A to G) ===")
    for node, c, fr in trace:
        print(f"process {node} (g={c}) | frontier: {fr}")
    print("UCS path :", " -> ".join(path), "| edges =", len(path) - 1, "| cost =", cost)
    print("UCS order:", order)
    print("Independent cost check:", path_cost(graph, path))

    bpath, border = bfs(graph, "A", "G")
    print("\n=== BFS on the same weighted topology ===")
    print("BFS path :", " -> ".join(bpath), "| edges =", len(bpath) - 1,
          "| cost =", path_cost(graph, bpath))
    print("BFS order:", border)

    print("\n=== Edge-case tests ===")
    p, c, o, _ = ucs(graph, "A", "A")
    print("start==goal  :", p, c)
    g2 = dict(graph); g2["H"] = []
    print("unreachable  :", ucs(g2, "A", "H")[:2])
    cyc = {"A": [("B", 2)], "B": [("A", 1), ("C", 2)], "C": []}
    print("cycle graph  :", ucs(cyc, "A", "C")[:2])
    try:
        ucs({"A": [("B", -2)], "B": []}, "A", "B")
    except ValueError as e:
        print("negative edge:", e)