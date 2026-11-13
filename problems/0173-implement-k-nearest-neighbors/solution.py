import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    # GUARD
    if not points or k <= 0:
        return []
    k = min(k, len(points))
    P = np.array(points); q = np.array(query_point)
    # DISTANCE
    d = np.sqrt(np.sum((P - q) ** 2, axis=1))
    # STABLE SORT
    idx = np.argsort(d, kind='stable')[:k]
    # TAKE K
    return [tuple(P[i]) for i in idx]