'''
  File name: edgeLink.py
  Author: Tarmily Wen
  Date created: Dec. 8, 2019
'''

import numpy as np
from collections import deque


def edgeLink(M, Mag, edge_Ori):
    H, W = Mag.shape
    linked = np.zeros_like(Mag)

    # Adaptive threshold selection: find a low threshold >= 4
    for pct in [70, 75, 80, 85, 90, 96]:
        low = np.percentile(Mag, pct)
        if low >= 4:
            break

    if low < 30:
        high = 2 * low
    else:
        high = 1.755 * low

    print((low, high))

    # Mark strong edges (above high threshold and passed NMS)
    strong = np.zeros((H, W), dtype=bool)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if Mag[y, x] >= high and M[y, x] == 1:
                linked[y, x] = 1
                strong[y, x] = True

    # BFS to connect weak edges (between low and high) to strong edges
    queue = deque(zip(*np.where(strong)))
    weak = (Mag >= low) & (M == 1)

    while queue:
        y, x = queue.popleft()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if weak[ny, nx] and linked[ny, nx] == 0:
                        linked[ny, nx] = 1
                        queue.append((ny, nx))

    return linked
