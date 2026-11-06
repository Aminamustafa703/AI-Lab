import itertools

INF = float("inf")


def build(leaves):
    """Flat leaf list -> nested [left, right] tree (complete binary)."""
    level = list(leaves)
    while len(level) > 1:
        level = [[level[i], level[i + 1]] for i in range(0, len(level), 2)]
    return level[0]


def leaves_of(t):
    return [t] if not isinstance(t, list) else leaves_of(t[0]) + leaves_of(t[1])


def minimax(t, max_turn, seen):
    if not isinstance(t, list):
        seen.append(t)
        return t
    vals = [minimax(c, not max_turn, seen) for c in t]
    return max(vals) if max_turn else min(vals)


def alphabeta(t, max_turn, alpha, beta, seen):
    if not isinstance(t, list):
        seen.append(t)
        return t
    value = -INF if max_turn else INF
    for c in t:
        v = alphabeta(c, not max_turn, alpha, beta, seen)
        if max_turn:
            value = max(value, v); alpha = max(alpha, value)
        else:
            value = min(value, v); beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def value_of(t, max_turn):
    return minimax(t, max_turn, [])


def order(t, max_turn, best=True):
    """Best ordering: MAX tries larger children first, MIN tries smaller first."""
    if not isinstance(t, list):
        return t
    kids = [order(c, not max_turn, best) for c in t]
    kids.sort(key=lambda c: value_of(c, not max_turn), reverse=(max_turn == best))
    return kids


def swapped(t, mask, i=0):
    """Swap the children of internal node i (heap numbering) when its mask bit is 1."""
    if not isinstance(t, list):
        return t
    a, b = swapped(t[0], mask, 2 * i + 1), swapped(t[1], mask, 2 * i + 2)
    return [b, a] if (mask >> i) & 1 else [a, b]


def run(name, tree):
    seen_mm, seen_ab = [], []
    v_mm = minimax(tree, True, seen_mm)
    v_ab = alphabeta(tree, True, -INF, INF, seen_ab)
    print(f"{name}: leaves = {leaves_of(tree)}")
    print(f"    Minimax = {v_mm} ({len(seen_mm)} leaves) | Alpha-beta = {v_ab} ({len(seen_ab)} leaves)")
    print(f"    alpha-beta evaluation order: {seen_ab}")
    return v_mm, v_ab, len(seen_ab)


def sweep(name, leaves):
    base = build(leaves)
    counts, values = [], set()
    for mask in range(2 ** 7):                       
        v_mm, v_ab, n = run_quiet(swapped(base, mask))
        counts.append(n); values.add((v_mm, v_ab))
    print(f"{name}: {len(counts)} orderings, root values seen = {sorted(values)}")
    print(f"    leaves evaluated: min = {min(counts)}, max = {max(counts)}, "
          f"mean = {sum(counts) / len(counts):.2f}")


def run_quiet(tree):
    a, b = [], []
    return minimax(tree, True, a), alphabeta(tree, True, -INF, INF, b), len(b)


if __name__ == "__main__":
    leaves = [3, 5, 2, 9, 3, 5, 2, 9]                
    base = build(leaves)
    print("=== Named orderings of Tree B ===")
    run("original", base)
    run("mirrored (all swapped)", swapped(base, 0b1111111))
    run("best-first ordering", order(base, True, best=True))
    run("worst-first ordering", order(base, True, best=False))

    print("\n=== Every combination of child swaps (2^7 = 128 orderings) ===")
    sweep("Tree B [3,5,2,9,3,5,2,9]", leaves)
    sweep("Tree A [3,5,2,9,0,7,0,0]", [3, 5, 2, 9, 0, 7, 0, 0])