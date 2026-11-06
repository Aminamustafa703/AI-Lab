
graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": ["G"],
    "F": [],
    "G": []
}

def dls(start, goal, limit):
    """
    Returns: (status, path)
    status → "FOUND", "CUTOFF", or "FAILURE"
    """
    def recursive_dls(node, path, depth):
        if node == goal:
            return "FOUND", path

        if depth == limit:
            if any(neighbor not in path for neighbor in graph[node]):
                return "CUTOFF", None
            else:
                return "FAILURE", None

        cutoff_occurred = False

        for neighbor in graph[node]:
            if neighbor not in path:          # Path-local cycle check
                status, result = recursive_dls(neighbor, path + [neighbor], depth + 1)

                if status == "FOUND":
                    return "FOUND", result
                elif status == "CUTOFF":
                    cutoff_occurred = True

        if cutoff_occurred:
            return "CUTOFF", None
        else:
            return "FAILURE", None

    return recursive_dls(start, [start], 0)


def iddfs(start, goal, max_depth):
    """
    Tries increasing depth limits from 0 to max_depth
    """
    for depth in range(max_depth + 1):
        status, path = dls(start, goal, depth)
        print(f"  Limit {depth}: {status}", end="")
        if status == "FOUND":
            print(f" → Path: {path}")
            return status, path, depth
        else:
            print()
    
    return "FAILURE", None, max_depth



print("=" * 55)
print("DLS Results (Limits 1 to 4)")
print("=" * 55)

for limit in [1, 2, 3, 4]:
    status, path = dls("A", "G", limit)
    print(f"Limit {limit}: {status}", end="")
    if status == "FOUND":
        print(f" → Path: {path}")
    else:
        print()

print("\n" + "=" * 55)
print("IDDFS Results (Max Depths 1 to 4)")
print("=" * 55)

for max_d in [1, 2, 3, 4]:
    print(f"\n--- IDDFS with max_depth = {max_d} ---")
    status, path, used = iddfs("A", "G", max_d)
    if status == "FOUND":
        print(f"→ Success at depth {used}")
        print(f"→ Path: {path}")
    else:
        print("→ FAILURE")
