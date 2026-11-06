
import math


def minimax(depth, index, max_turn, scores, target, seen):
    """Minimax; every leaf evaluation is recorded in `seen`."""
    if depth == target:
        seen.append(index)                    
        return scores[index]
    vals = [minimax(depth + 1, index * 2 + i, not max_turn, scores, target, seen)
            for i in range(2)]
    return max(vals) if max_turn else min(vals)


def alphabeta(depth, index, max_turn, scores, target, alpha, beta, seen, cuts):
    """Alpha-beta; leaf evaluations go in `seen`, pruning decisions in `cuts`."""
    if depth == target:
        seen.append(index)                    
        return scores[index]
    if max_turn:
        value = float("-inf")
        for i in range(2):
            value = max(value, alphabeta(depth + 1, index * 2 + i, False, scores,
                                         target, alpha, beta, seen, cuts))
            alpha = max(alpha, value)
            if alpha >= beta:
                if i == 0:
                    cuts.append(f"cut at MAX (d{depth},#{index}): alpha={alpha} >= beta={beta}")
                break
        return value
    value = float("inf")
    for i in range(2):
        value = min(value, alphabeta(depth + 1, index * 2 + i, True, scores,
                                     target, alpha, beta, seen, cuts))
        beta = min(beta, value)
        if alpha >= beta:
            if i == 0:
                cuts.append(f"cut at MIN (d{depth},#{index}): alpha={alpha} >= beta={beta}")
            break
    return value


def compare(name, scores):
    depth = int(math.log2(len(scores)))
    seen_mm, seen_ab, cuts = [], [], []
    v_mm = minimax(0, 0, True, scores, depth, seen_mm)
    v_ab = alphabeta(0, 0, True, scores, depth, float("-inf"), float("inf"), seen_ab, cuts)
    print(f"\n=== {name}: leaves = {scores} ===")
    print(f"Minimax    : root = {v_mm}, leaves evaluated = {len(seen_mm)}")
    print(f"Alpha-beta : root = {v_ab}, leaves evaluated = {len(seen_ab)}, order = {seen_ab}")
    for c in cuts:
        print("  ", c)
    skipped = sorted(set(seen_mm) - set(seen_ab))
    print(f"Leaves skipped by pruning (index): {skipped or 'none'}")
    print("Same root value:", v_mm == v_ab, "| leaves saved:", len(seen_mm) - len(seen_ab))
    return v_mm, v_ab


if __name__ == "__main__":
    compare("Tree A (Section 5.1, padded)", [3, 5, 2, 9, 0, 7, 0, 0])
    compare("Tree B (Section 5.2)", [3, 5, 2, 9, 3, 5, 2, 9])
    compare("Edge case: all leaves equal", [4] * 8)
    compare("Edge case: single leaf", [7])