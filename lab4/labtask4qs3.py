import heapq

graph = {"S": [("A", 1), ("B", 4)], "A": [("C", 2)],
         "B": [("G", 5)], "C": [("G", 3)], "G": []}
h_original = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}
true_remaining = {"S": 6, "A": 5, "B": 5, "C": 3, "G": 0}


def search(g, h, start, goal):
    """A*: entry = (f, g, node, path). Expansion = successors examined
    (goal and stale entries are not counted)."""
    pq = [(h[start], 0, start, [start])]
    best, log, expansions = {start: 0}, [], 0
    while pq:
        f, cost, node, path = heapq.heappop(pq)
        if cost != best[node]:
            continue
        if node == goal:
            log.append(f"process {node}: g={cost}, h={h[node]}, f={f} -> GOAL")
            return path, cost, log, expansions
        expansions += 1
        log.append(f"expand  {node}: g={cost}, h={h[node]}, f={f}")
        for nxt, w in g[node]:
            ng = cost + w
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                heapq.heappush(pq, (ng + h[nxt], ng, nxt, path + [nxt]))
        fr = ", ".join(f"{n}(f={ff})" for ff, _, n, _ in sorted(pq))
        log.append(f"        frontier: {fr}")
    return None, float("inf"), log, expansions


def path_cost(g, path):
    return sum(dict(g[u])[v] for u, v in zip(path, path[1:]))


def audit(h):
    bad_adm = [n for n in graph if h[n] > true_remaining[n]]
    bad_con = [(u, v) for u, es in graph.items() for v, w in es if h[u] > w + h[v]]
    return bad_adm, bad_con


def run(title, h):
    path, cost, log, exp = search(graph, h, "S", "G")
    print(f"\n=== {title} ===")
    for line in log:
        print(line)
    print("path:", " -> ".join(path), "| returned cost:", cost,
          "| independent sum:", path_cost(graph, path), "| expanded:", exp)
    bad_adm, bad_con = audit(h)
    print("overestimating nodes:", bad_adm or "none", "| inconsistent edges:", bad_con or "none")
    return path, cost, exp


if __name__ == "__main__":
    print("Changed value: h(C) from", h_original["C"], "to 10 (true remaining cost of C =",
          true_remaining["C"], ")")
    po, co, eo = run("Original heuristic (h(C) = 2)", h_original)
    h_bad = dict(h_original); h_bad["C"] = 10        
    pm, cm, em = run("Modified heuristic (h(C) = 10)", h_bad)

    print("\n=== Comparison ===")
    print(f"{'Run':<10}{'h(C)':<6}{'Path':<18}{'Cost':<6}{'Expanded'}")
    print(f"{'original':<10}{2:<6}{' -> '.join(po):<18}{co:<6}{eo}")
    print(f"{'modified':<10}{10:<6}{' -> '.join(pm):<18}{cm:<6}{em}")

    h_mild = dict(h_original); h_mild["C"] = 4
    run("Extra test: mild overestimate h(C) = 4", h_mild)

    print("\n=== Sweep: only h(C) changes (true remaining cost of C = 3) ===")
    print(f"{'h(C)':<6}{'Admissible':<12}{'Path':<18}{'Cost':<6}{'Expanded'}")
    for v in range(2, 10):
        hh = dict(h_original); hh["C"] = v
        p, c, _, e = search(graph, hh, "S", "G")
        print(f"{v:<6}{'yes' if v <= 3 else 'NO':<12}{' -> '.join(p):<18}{c:<6}{e}")