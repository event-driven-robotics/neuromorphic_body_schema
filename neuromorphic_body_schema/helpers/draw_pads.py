"""
Implementetation is based on the code written in */icub-main/src/tools/iCubSkinGui/plugin/include/*
"""
import math


def triangle(cx, cy, th, lr_mirror=0):
    DEG2RAD = math.pi / 180.0

    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    H = math.sin(DEG2RAD * 60.0)
    L = 2.0 * H / 9.0

    nVerts = 3
    nTaxels = 12

    dX = [0] * 12
    dY = [0] * 12

    dX[5] = L * math.cos(DEG2RAD * 30.0)
    dX[4] = 0.5 - dX[5]
    dX[2] = 0.5 + dX[5]
    dX[1] = 1.0 - dX[5]
    dX[6] = 0.25
    dX[3] = 0.5
    dX[0] = 0.75
    dX[7] = 0.25 + dX[5]
    dX[11] = 0.75 - dX[5]
    dX[8] = dX[7]
    dX[10] = dX[11]
    dX[9] = 0.5

    dY[5] = L * math.sin(DEG2RAD * 30.0)
    dY[4] = dY[5]
    dY[2] = dY[5]
    dY[1] = dY[5]
    dY[6] = 0.5 * H - L
    dY[3] = L
    dY[0] = dY[6]
    dY[7] = 0.5 * H - math.sin(DEG2RAD * 30.0) * L
    dY[11] = dY[7]
    dY[8] = 0.5 * H + math.sin(DEG2RAD * 30.0) * L
    dY[10] = dY[8]
    dY[9] = H - L

    for i in range(nTaxels):
        x = 30.0 * dX[i] - 15.0
        y = 30.0 * dY[i] - 10.0 * H

        if lr_mirror == 1:
            x = -x
        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    dXv = [-15.0, 15.0, 0]
    dYv = [-8.66, -8.66, 17.32]

    dXmin = float('inf')
    dYmin = float('inf')
    dXmax = float('-inf')
    dYmax = float('-inf')

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

        if dXv[i] < dXmin:
            dXmin = dXv[i]
        if dXv[i] > dXmax:
            dXmax = dXv[i]
        if dYv[i] < dYmin:
            dYmin = dYv[i]
        if dYv[i] > dYmax:
            dYmax = dYv[i]

    return dX, dY, dXv, dYv


def triangle_10pad(cx, cy, th, lr_mirror=0):
    DEG2RAD = math.pi / 180.0
    th = th
    CST = math.cos(DEG2RAD * (th + 0))
    SNT = math.sin(DEG2RAD * (th + 0))

    H = math.sin(DEG2RAD * 60.0)
    scale = 1 / 40.0
    nVerts = 3
    nTaxels = 12

    dX = [-128.62, 0, 0, 128.62, 257.2, -256.3,
          385.83, 128.62, 0.0, -128.62, -385.83, -257.2]
    dY = [222.83, 296.55, 445.54, 222.83, 0, -147.64,
          -222.83, -222.83, 0.0, -222.83, -222.83, 0.0]

    for i in range(nTaxels):
        dX[i] *= scale
        dY[i] *= scale

    dXv = [-55.0 / 4, 55.0 / 4, 0.0]
    dYv = [-32.5 / 4, -32.5 / 4, 63.0 / 4]

    for i in range(nTaxels):
        x = dX[i]
        y = dY[i]
        if lr_mirror == 1:
            x = -x

        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

    return dX, dY, dXv, dYv


def fingertip2L(cx, cy, th, lr_mirror=0):
    nVerts = 7
    nTaxels = 12

    dX = [41.0, 15.0, 15.0, 41.0, 30.0, 11.0,
          0.0, -11.0, -30.0, -41.0, -15.0, -15.0]
    dY = [10.0, 10.0, 35.0, 35.0, 64.0, 58.0,
          82.0, 58.0, 64.0, 35.0, 35.0, 10.0]

    dXv = [53.0, 53.0, dX[4] + 10.0, 0.0, -(dX[4] + 10.0), -53.0, -53.0]
    dYv = [0.0, 45.0, dY[4] + 10.0, dY[6] + 12.0, dY[4] + 10.0, 45.0, 0.0]

    scale = 2.7 / 15.3
    for i in range(nTaxels):
        dX[i] *= scale
        dY[i] *= scale
    for i in range(nVerts):
        dXv[i] *= scale
        dYv[i] *= scale

    DEG2RAD = math.pi / 180.0
    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    for i in range(nTaxels):
        x = dX[i]
        y = dY[i]
        if lr_mirror == 1:
            x = -x

        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

    return dX, dY, dXv, dYv


def fingertip2R(cx, cy, th, lr_mirror=0):
    nVerts = 7
    nTaxels = 12

    dX = [-41.0, 15.0, 15.0, 41.0, 30.0, 11.0,
          0.0, -11.0, -30.0, -41.0, -15.0, -15.0]
    dY = [10.0, 10.0, 35.0, 35.0, 64.0, 58.0,
          82.0, 58.0, 64.0, 35.0, 35.0, 10.0]

    dXv = [53.0, 53.0, dX[3] + 10.0, 0.0, -(dX[3] + 10.0), -53.0, -53.0]
    dYv = [0.0, 45.0, dY[3] + 10.0, dY[5] + 12.0, dY[3] + 10.0, 45.0, 0.0]

    scale = 2.7 / 15.3
    for i in range(nTaxels):
        dX[i] *= scale
        dY[i] *= scale
    for i in range(nVerts):
        dXv[i] *= scale
        dYv[i] *= scale

    DEG2RAD = math.pi / 180.0
    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    for i in range(nTaxels):
        x = dX[i]
        y = dY[i]
        if lr_mirror == 1:
            x = -x

        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

    return dX, dY, dXv, dYv


def fingertip3L(cx, cy, th, lr_mirror=0):
    nVerts = 7
    nTaxels = 12

    dX = [30.0, 0.0, -30.0, -45.0, 45.0, -
          30.0, 0.0, 30.0, 25.0, 0.0, -25.0, 0.0]
    dY = [-20.0, -20.0, -20.0, -39.0, -39.0,
          -58.0, -58.0, -58.0, 10.0, 0.0, 10.0, 40.0]

    dXv = [53.0, 53.0, dX[4] + 10.0, 0.0, -(dX[4] + 10.0), -53.0, -53.0]
    dYv = [0.0, 45.0, dY[4] + 10.0, dY[6] + 12.0, dY[4] + 10.0, 45.0, 0.0]

    scale = 2.7 / 15.3
    for i in range(nTaxels):
        dX[i] *= scale
        dY[i] *= scale
    for i in range(nVerts):
        dXv[i] *= scale
        dYv[i] *= scale

    DEG2RAD = math.pi / 180.0
    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    for i in range(nTaxels):
        x = dX[i]
        y = dY[i]
        if lr_mirror == 1:
            x = -x

        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        dXv[i] = 0
        dYv[i] = 0

    return dX, dY, dXv, dYv


def fingertip3R(cx, cy, th, lr_mirror=0):
    nVerts = 7
    nTaxels = 12

    dX = [-30.0, 0.0, 30.0, 45.0, -45.0, 30.0,
          0.0, -30.0, -25.0, 0.0, 25.0, 0.0]
    dY = [-20.0, -20.0, -20.0, -39.0, -39.0,
          -58.0, -58.0, -58.0, 10.0, 0.0, 10.0, 40.0]

    dXv = [53.0, 53.0, dX[3] + 10.0, 0.0, -(dX[3] + 10.0), -53.0, -53.0]
    dYv = [0.0, 45.0, dY[3] + 10.0, dY[5] + 12.0, dY[3] + 10.0, 45.0, 0.0]

    scale = 2.7 / 15.3
    for i in range(nTaxels):
        dX[i] *= scale
        dY[i] *= scale
    for i in range(nVerts):
        dXv[i] *= scale
        dYv[i] *= scale

    DEG2RAD = math.pi / 180.0
    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    for i in range(nTaxels):
        x = dX[i]
        y = dY[i]
        if lr_mirror == 1:
            x = -x

        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x

        # TODO fill this with numbers if triangles wanted
        dXv[i] = 0
        dYv[i] = 0

    return dX, dY, dXv, dYv


def palmR(cx, cy, th, lr_mirror=0):
    DEG2RAD = math.pi / 180.0

    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    nVerts = 4
    nTaxels = 48
    m_RadiusOrig = 1.8

    dX = [0] * nTaxels
    dY = [0] * nTaxels

    dX[27], dY[27] = 1.5, 6.5
    dX[26], dY[26] = 6.5, 6
    dX[25], dY[25] = 11.5, 6
    dX[24], dY[24] = 16.5, 6
    dX[31], dY[31] = 21.5, 6
    dX[29], dY[29] = 6.5, 1
    dX[28], dY[28] = 11.5, 1
    dX[32], dY[32] = 16.5, 1
    dX[33], dY[33] = 21.5, 1
    dX[35], dY[35] = 9.5, -2
    dX[30], dY[30] = 14.5, -3.5
    dX[34], dY[34] = 21.5, -4

    dX[6], dY[6] = 27, 6
    dX[3], dY[3] = 32, 6
    dX[2], dY[2] = 37, 6
    dX[1], dY[1] = 42, 5.5
    dX[0], dY[0] = 47, 4.5
    dX[11], dY[11] = 51.7, 4
    dX[7], dY[7] = 27, 1
    dX[8], dY[8] = 32, 1
    dX[4], dY[4] = 37, 1
    dX[5], dY[5] = 42, 0
    dX[9], dY[9] = 27, -3.5
    dX[10], dY[10] = 32, -3.5

    dX[16], dY[16] = 37.5, -4.5
    dX[15], dY[15] = 42, -5.5
    dX[14], dY[14] = 46.5, -8
    dX[20], dY[20] = 27, -9
    dX[21], dY[21] = 32, -9
    dX[17], dY[17] = 37, -9
    dX[19], dY[19] = 42, -10.5
    dX[22], dY[22] = 38, -14
    dX[18], dY[18] = 43, -16
    dX[13], dY[13] = 47, -13
    dX[12], dY[12] = 47.5, -18
    dX[23], dY[23] = 43.5, -20

    dX[46], dY[46] = 33, -14.5
    dX[47], dY[47] = 28, -14.5
    dX[36], dY[36] = 28, -19.5
    dX[42], dY[42] = 33, -19.5
    dX[45], dY[45] = 38, -19.5
    dX[37], dY[37] = 28, -24.5
    dX[38], dY[38] = 33, -24.5
    dX[41], dY[41] = 38, -24.5
    dX[44], dY[44] = 43, -26
    dX[39], dY[39] = 35, -29
    dX[40], dY[40] = 40, -29.5
    dX[43], dY[43] = 37, -32.5

    for i in range(nTaxels):
        x = 1.2 * dX[i]
        y = 1.2 * dY[i]

        if lr_mirror == 1:
            x = -x
        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    dXv = [-25, 50, 50, -25]
    dYv = [-50, -50, 50, 50]

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x
        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

    return dX, dY, dXv, dYv


def palmL(cx, cy, th, lr_mirror=0):
    DEG2RAD = math.pi / 180.0

    CST = math.cos(DEG2RAD * th)
    SNT = math.sin(DEG2RAD * th)

    H = math.sin(DEG2RAD * 60.0)
    L = 2.0 * H / 9.0

    nVerts = 4
    nTaxels = 48
    m_RadiusOrig = 1.8

    dX = [0.0] * nTaxels
    dY = [0.0] * nTaxels

    dX[29], dY[29] = 1.5, 6.5
    dX[28], dY[28] = 6.5, 6
    dX[30], dY[30] = 11.5, 6
    dX[31], dY[31] = 16.5, 6
    dX[33], dY[33] = 21.5, 6
    dX[27], dY[27] = 6.5, 1
    dX[26], dY[26] = 11.5, 1
    dX[32], dY[32] = 16.5, 1
    dX[34], dY[34] = 21.5, 1
    dX[35], dY[35] = 9.5, -2  # thermal_pad
    dX[25], dY[25] = 14.5, -3.5
    dX[24], dY[24] = 21.5, -4

    dX[6], dY[6] = 27, 6
    dX[7], dY[7] = 32, 6
    dX[8], dY[8] = 37, 6
    dX[9], dY[9] = 42, 5.5
    dX[10], dY[10] = 47, 4.5
    dX[11], dY[11] = 51.7, 4  # thermal_pad
    dX[3], dY[3] = 27, 1
    dX[1], dY[1] = 32, 1
    dX[4], dY[4] = 37, 1
    dX[5], dY[5] = 42, 0
    dX[2], dY[2] = 27, -3.5
    dX[0], dY[0] = 32, -3.5

    dX[17], dY[17] = 37.5, -4.5
    dX[20], dY[20] = 42, -5.5
    dX[21], dY[21] = 46.5, -8
    dX[15], dY[15] = 27, -9
    dX[14], dY[14] = 32, -9
    dX[16], dY[16] = 37, -9
    dX[19], dY[19] = 42, -10.5

    dX[13], dY[13] = 38, -14
    dX[18], dY[18] = 43, -16
    dX[22], dY[22] = 47, -13
    dX[12], dY[12] = 47.5, -18
    dX[23], dY[23] = 43.5, -20  # thermal_pad

    dX[45], dY[45] = 28, -19.5
    dX[42], dY[42] = 33, -19.5
    dX[36], dY[36] = 38, -19.5
    dX[44], dY[44] = 28, -24.5
    dX[40], dY[40] = 33, -24.5
    dX[38], dY[38] = 38, -24.5
    dX[37], dY[37] = 43, -26
    dX[41], dY[41] = 35, -29
    dX[39], dY[39] = 40, -29.5
    dX[43], dY[43] = 37, -32.5  # thermal pad
    dX[47], dY[47] = 33, -14.5
    dX[46], dY[46] = 28, -14.5

    for i in range(nTaxels):
        x = 1.2 * dX[i] - 0.0
        y = 1.2 * dY[i] - 0.0

        if lr_mirror == 1:
            x = -x
        dX[i] = cx + CST * x - SNT * y
        dY[i] = cy + SNT * x + CST * y

    dXv = [-25, 50, 50, -25]
    dYv = [-50, -50, 50, 50]

    for i in range(nVerts):
        x = dXv[i]
        y = dYv[i]
        if lr_mirror == 1:
            x = -x
        dXv[i] = cx + CST * x - SNT * y
        dYv[i] = cy + SNT * x + CST * y

    return dX, dY, dXv, dYv
