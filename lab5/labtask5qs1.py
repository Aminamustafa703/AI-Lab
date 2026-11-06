
import math


def minimax(cur_depth, node_index, max_turn, scores, target_depth):
    """Supplied minimax (Section 5.1)."""
    if cur_depth == target_depth:
        return scores[node_index]
    if max_turn:
        return max(minimax(cur_depth + 1, node_index * 2, False, scores, target_depth),
                   minimax(cur_depth + 1, node_index * 2 + 1, False, scores, target_depth))
    return min(minimax(cur_depth + 1, node_index * 2, True, scores, target_depth),
               minimax(cur_depth + 1, node_index * 2 + 1, True, scores, target_depth))

def minimax_trace(depth, index, max_turn, scores, target, log):
    """Same minimax, but records every node value in the order it is computed."""
    if depth == target:
        log.append(f"leaf  (d{depth},#{index}) = {scores[index]}")
        return scores[index]
    left = minimax_trace(depth + 1, index * 2, not max_turn, scores, target, log)
    right = minimax_trace(depth + 1, index * 2 + 1, not max_turn, scores, target, log)
    op = "max" if max_turn else "min"
    value = max(left, right) if max_turn else min(left, right)
    log.append(f"{'MAX' if max_turn else 'MIN'}   (d{depth},#{index}) = {op}({left}, {right}) = {value}")
    return value


def padded(raw, pad=0):
    depth = math.ceil(math.log2(len(raw)))
    return raw + [pad] * (2 ** depth - len(raw)), depth


if __name__ == "__main__":
    raw = [3, 5, 2, 9, 0, 7]
    scores, depth = padded(list(raw))
    print("Supplied scores:", raw)
    print("Padded leaves  :", scores, "| tree depth =", depth)

    print("\n=== Bottom-up trace (order of computation) ===")
    log = []
    root = minimax_trace(0, 0, True, scores, depth, log)
    for line in log:
        print(line)
    print("\nRoot value (trace)   :", root)
    print("Root value (supplied):", minimax(0, 0, True, scores, depth))

    left = minimax(1, 0, False, scores, depth)
    right = minimax(1, 1, False, scores, depth)
    print(f"MAX chooses: left child = {left}, right child = {right} ->",
          "LEFT" if left >= right else "RIGHT")

    print("\n=== Edge case: effect of the padding value ===")
    for pad in (0, 7, 9):
        s, d = padded(list(raw), pad)
        print(f"pad={pad}: leaves={s} -> root value = {minimax(0, 0, True, s, d)}")