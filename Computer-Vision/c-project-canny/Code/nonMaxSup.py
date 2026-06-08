'''
  File name: nonMaxSup.py
  Author: Tarmily Wen
  Date created: Dec. 8, 2019
'''

import numpy as np

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''



def nonMaxSup(Mag, Ori, grad_Ori):
    ###############################################################################
    # For each pixel, compare magnitude against the two neighbors along the
    # gradient direction. Keep only local maxima (set to 1); suppress others.
    # grad_Ori is quantized to {0, ±45, ±90, ±135, ±180} degrees.
    ###############################################################################
    H, W = Mag.shape
    suppressed = np.zeros((H, W), dtype=np.float64)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            angle = grad_Ori[y, x]
            mag = Mag[y, x]

            if angle in (0, 180, -180):
                n1 = Mag[y, x - 1]
                n2 = Mag[y, x + 1]
            elif angle in (90, -90):
                n1 = Mag[y - 1, x]
                n2 = Mag[y + 1, x]
            elif angle in (45, -135):
                # gradient points ↗ / ↙ : compare (y-1,x+1) and (y+1,x-1)
                n1 = Mag[y - 1, x + 1]
                n2 = Mag[y + 1, x - 1]
            else:
                # angle in (135, -45): gradient points ↘ / ↖
                n1 = Mag[y - 1, x - 1]
                n2 = Mag[y + 1, x + 1]

            if mag >= n1 and mag >= n2:
                suppressed[y, x] = 1

    return suppressed

