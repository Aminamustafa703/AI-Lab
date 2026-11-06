
import heapq

graph = {"S": [("A", 1), ("B", 4)], "A": [("C", 2)],
         "B": [("G", 5)], "C": [("G", 3)], "G": []}
h_astar = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}
h_zero = {n: 0 for n in graph}            

def search(g, h, start, goal):
    """Same priority-queue code for A* and UCS; only h differs.
    Entry = (f, g, node, path); ties: smaller g, then node name.
    Expansion = successors examined (goal and stale entries are not counted)."""
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    log, expansions, stale = [], 0, 0
    while pq:
        f, cost, node, path = heapq.heappop(pq)
        if cost != best[node]:
            stale += 1                      
            continue
        if node == goal:
            log.append(f"process {node}: g={cost}, h={h[node]}, f={f} -> GOAL (not counted)")
            return path, cost, log, expansions, stale
        expansions += 1
        log.append(f"expand  {node}: g={cost}, h={h[node]}, f={f}")
        for nxt, w in g.get(node, []):
            ng = cost + w
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(pq, (ng + h[nxt], ng, nxt, path + [nxt]))
        fr = ", ".join(f"{n}(f={ff})" for ff, _, n, _ in sorted(pq))
        log.append(f"        frontier: {fr}")
    return None, float("inf"), log, expansions, stale


def path_cost(g, path):
    """Independent check: add the edge weights along the path."""
    return sum(dict(g[u])[v] for u, v in zip(path, path[1:]))


def report(title, h):
    path, cost, log, exp, stale = search(graph, h, "S", "G")
    print(f"\n=== {title} ===")
    for line in log:
        print(line)
    print("path           :", " -> ".join(path))
    print("returned cost  :", cost, "| independent sum:", path_cost(graph, path))
    print("expanded nodes :", exp, "| stale entries skipped:", stale)
    return path, cost, exp


if __name__ == "__main__":
    pa, ca, ea = report("A* (supplied heuristic)", h_astar)
    pu, cu, eu = report("UCS (h = 0 for all nodes)", h_zero)

    print("\n=== Comparison ===")
    print(f"{'Run':<6}{'Path':<18}{'Cost':<6}{'Expanded':<10}")
    print(f"{'A*':<6}{' -> '.join(pa):<18}{ca:<6}{ea:<10}")
    print(f"{'UCS':<6}{' -> '.join(pu):<18}{cu:<6}{eu:<10}")

    print("\n=== Edge-case tests ===")
    r = search(graph, h_astar, "S", "S")
    print("start==goal  : path", r[0], "cost", r[1], "expanded", r[3])
    g2 = {**graph, "X": []}
    r = search(g2, {**h_astar, "X": 0}, "S", "X")
    print("unreachable X: path", r[0], "expanded", r[3])