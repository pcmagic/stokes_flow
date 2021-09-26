import numpy as np

Ra = 1

unode = np.arange(3)
fnodes = np.arange(12).reshape(4, 3) ** 2
img_fnodes = np.arange(12).reshape(4, 3) ** 0.5

dxi = (unode - fnodes).T
disx = dxi[0]
disy = dxi[1]
disz = dxi[2]
r2sk = disx ** 2 + disy ** 2 + disz ** 2
rsk = r2sk ** 0.5
r3sk = r2sk ** 1.5

H1sk = 1.0 / r3sk
MSxx = (r2sk + disx ** 2) * H1sk
MSyx = disy * disx * H1sk
MSxy = MSyx
MSyy = (r2sk + disy ** 2) * H1sk
MSxz = disx * disz * H1sk
MSyz = disy * disz * H1sk
MSzx = MSxz
MSzy = MSyz
MSzz = (r2sk + disz ** 2) * H1sk

img_dxi = (unode - img_fnodes).T
disxi = img_dxi[0]
disyi = img_dxi[1]
diszi = img_dxi[2]
r2ski = disxi ** 2 + disyi ** 2 + diszi ** 2
rski = r2ski ** 0.5
r3ski = r2ski ** 1.5
r5ski = r2ski ** 2.5
H1ski = 1 / rski
H2ski = 1 / r2ski
H3ski = 1 / r3ski
H5ski = 1 / r5ski

Xf = np.linalg.norm(fnodes, axis=1)
Xfi = np.linalg.norm(img_fnodes, axis=1)
xf = np.linalg.norm(unode)
Dx1 = np.sum(img_dxi.T * img_fnodes, axis=1)
Dx2 = np.sum(unode * img_fnodes, axis=1)

A = 0.5 * (Xf ** 2 - Ra ** 2) / Xf ** 3
B = r2ski * (rski - Xfi) * Xfi
C = 3 * Ra / Xfi
Det = 1 / (Xfi * (Xfi * rski + Dx2 - Xfi ** 2))
E = 1 / (xf * Xfi * (xf * Xfi + Dx2))

A1 = (Xf ** 2 - Ra ** 2) / Xf
A2 = xf ** 2 - Ra ** 2

Pxx = A * (-3 * fnodes[:, 0] * disxi * H3ski / Ra + Ra * H3ski - 3 * Ra * disxi ** 2 * H5ski
           - 2 * fnodes[:, 0] * img_fnodes[:, 0] * H3ski / Ra
           + 6 * fnodes[:, 0] * H5ski * disxi * Dx1 / Ra
           + C * (img_fnodes[:, 0] * disxi * r2ski + disxi ** 2 * Xfi ** 2 + B)
           * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (Xfi * disxi + img_fnodes[:, 0] * rski)
           * (img_fnodes[:, 0] * r2ski - disxi * Xfi ** 2 + (unode[0] - 2 * img_fnodes[:, 0])
              * rski * Xfi) - C * E * (unode[0] * img_fnodes[:, 0] + xf * Xfi)
           + C * E ** 2 * (Xfi * unode[0] + xf * img_fnodes[:, 0]) * (
                   Xfi * unode[0] + xf * img_fnodes[:, 0]) * xf * Xfi)
Pyy = A * (-3 * fnodes[:, 1] * disyi * H3ski / Ra + Ra * H3ski - 3 * Ra * disyi ** 2 * H5ski
           - 2 * fnodes[:, 1] * img_fnodes[:, 1] * H3ski / Ra
           + 6 * fnodes[:, 1] * H5ski * disyi * Dx1 / Ra
           + C * (img_fnodes[:, 1] * disyi * r2ski + disyi ** 2 * Xfi ** 2 + B)
           * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (Xfi * disyi + img_fnodes[:, 1] * rski)
           * (img_fnodes[:, 1] * r2ski - disyi * Xfi ** 2
              + (unode[1] - 2 * img_fnodes[:, 1]) * rski * Xfi) - C * E * (
                   unode[1] * img_fnodes[:, 1] + xf * Xfi) + C * E ** 2 * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * xf * Xfi)
Pzz = A * (-3 * fnodes[:,
                2] * diszi * H3ski / Ra + Ra * H3ski - 3 * Ra * diszi ** 2 * H5ski - 2 * fnodes[:,
                                                                                         2] * img_fnodes[
                                                                                              :,
                                                                                              2] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    2] * H5ski * diszi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   2] * diszi * r2ski + diszi ** 2 * Xfi ** 2 + B) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * diszi + img_fnodes[:, 2] * rski) * (
                   img_fnodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                   unode[2] - 2 * img_fnodes[:, 2]) * rski * Xfi) - C * E * (
                   unode[2] * img_fnodes[:, 2] + xf * Xfi) + C * E ** 2 * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * xf * Xfi)

Pxy = A * (-3 * fnodes[:, 1] * disxi * H3ski / Ra - 3 * Ra * disxi * disyi * H5ski - 2 * fnodes[:,
                                                                                         1] * img_fnodes[
                                                                                              :,
                                                                                              0] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    1] * H5ski * disxi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   1] * disxi * r2ski + disxi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * disxi + img_fnodes[:, 0] * rski) * (
                   img_fnodes[:, 1] * r2ski - disyi * Xfi ** 2 + (
                   unode[1] - 2 * img_fnodes[:, 1]) * rski * Xfi) - C * E * (
                   unode[0] * img_fnodes[:, 1]) + C * E ** 2 * (
                   Xfi * unode[0] + xf * img_fnodes[:, 0]) * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * xf * Xfi)
Pyx = A * (-3 * fnodes[:, 0] * disyi * H3ski / Ra - 3 * Ra * disxi * disyi * H5ski - 2 * fnodes[:,
                                                                                         1] * img_fnodes[
                                                                                              :,
                                                                                              0] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    0] * H5ski * disyi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   0] * disyi * r2ski + disxi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * disyi + img_fnodes[:, 1] * rski) * (
                   img_fnodes[:, 0] * r2ski - disxi * Xfi ** 2 + (
                   unode[0] - 2 * img_fnodes[:, 0]) * rski * Xfi) - C * E * (
                   unode[1] * img_fnodes[:, 0]) + C * E ** 2 * (
                   Xfi * unode[0] + xf * img_fnodes[:, 0]) * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * xf * Xfi)

Pxz = A * (-3 * fnodes[:, 2] * disxi * H3ski / Ra - 3 * Ra * disxi * diszi * H5ski - 2 * fnodes[:,
                                                                                         2] * img_fnodes[
                                                                                              :,
                                                                                              0] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    2] * H5ski * disxi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   2] * disxi * r2ski + disxi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * disxi + img_fnodes[:, 0] * rski) * (
                   img_fnodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                   unode[2] - 2 * img_fnodes[:, 2]) * rski * Xfi) - C * E * (
                   unode[0] * img_fnodes[:, 2]) + C * E ** 2 * (
                   Xfi * unode[0] + xf * img_fnodes[:, 0]) * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * xf * Xfi)
Pzx = A * (-3 * fnodes[:, 0] * diszi * H3ski / Ra - 3 * Ra * disxi * diszi * H5ski - 2 * fnodes[:,
                                                                                         2] * img_fnodes[
                                                                                              :,
                                                                                              0] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    0] * H5ski * diszi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   0] * diszi * r2ski + disxi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * diszi + img_fnodes[:, 2] * rski) * (
                   img_fnodes[:, 0] * r2ski - disxi * Xfi ** 2 + (
                   unode[0] - 2 * img_fnodes[:, 0]) * rski * Xfi) - C * E * (
                   unode[2] * img_fnodes[:, 0]) + C * E ** 2 * (
                   Xfi * unode[0] + xf * img_fnodes[:, 0]) * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * xf * Xfi)

Pyz = A * (-3 * fnodes[:, 2] * disyi * H3ski / Ra - 3 * Ra * diszi * disyi * H5ski - 2 * fnodes[:,
                                                                                         1] * img_fnodes[
                                                                                              :,
                                                                                              2] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    2] * H5ski * disyi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   2] * disyi * r2ski + diszi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * disyi + img_fnodes[:, 1] * rski) * (
                   img_fnodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                   unode[2] - 2 * img_fnodes[:, 2]) * rski * Xfi) - C * E * (
                   unode[1] * img_fnodes[:, 2]) + C * E ** 2 * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * xf * Xfi)
Pzy = A * (-3 * fnodes[:, 1] * diszi * H3ski / Ra - 3 * Ra * disyi * diszi * H5ski - 2 * fnodes[:,
                                                                                         2] * img_fnodes[
                                                                                              :,
                                                                                              1] * H3ski / Ra + 6 * fnodes[
                                                                                                                    :,
                                                                                                                    1] * H5ski * diszi * Dx1 / Ra + C * (
                   img_fnodes[:,
                   1] * diszi * r2ski + disyi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                   Xfi * diszi + img_fnodes[:, 2] * rski) * (
                   img_fnodes[:, 1] * r2ski - disyi * Xfi ** 2 + (
                   unode[1] - 2 * img_fnodes[:, 1]) * rski * Xfi) - C * E * (
                   unode[2] * img_fnodes[:, 1]) + C * E ** 2 * (
                   Xfi * unode[1] + xf * img_fnodes[:, 1]) * (
                   Xfi * unode[2] + xf * img_fnodes[:, 2]) * xf * Xfi)

Mxx = MSxx - Ra * H1ski / Xf - Ra ** 3 * disxi * disxi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 0] * img_fnodes[:, 0] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 0] * disxi + img_fnodes[:, 0] * disxi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             0] * img_fnodes[:,
                                                                                  0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxx
Myy = MSyy - Ra * H1ski / Xf - Ra ** 3 * disyi * disyi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 1] * img_fnodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 1] * disyi + img_fnodes[:, 1] * disyi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             1] * img_fnodes[:,
                                                                                  1] * Dx1 * H3ski / Ra ** 3) - A2 * Pyy
Mzz = MSzz - Ra * H1ski / Xf - Ra ** 3 * diszi * diszi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 2] * img_fnodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 2] * diszi + img_fnodes[:, 2] * diszi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             2] * img_fnodes[:,
                                                                                  2] * Dx1 * H3ski / Ra ** 3) - A2 * Pzz

Mxy = MSxy - Ra ** 3 * disxi * disyi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 0] * img_fnodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 0] * disyi + img_fnodes[:, 1] * disxi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             1] * img_fnodes[:,
                                                                                  0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxy
Mxz = MSxz - Ra ** 3 * disxi * diszi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 0] * img_fnodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 0] * diszi + img_fnodes[:, 2] * disxi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             2] * img_fnodes[:,
                                                                                  0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxz
Myz = MSyz - Ra ** 3 * disyi * diszi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 2] * img_fnodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 2] * disyi + img_fnodes[:, 1] * diszi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             1] * img_fnodes[:,
                                                                                  2] * Dx1 * H3ski / Ra ** 3) - A2 * Pyz

Myx = MSyx - Ra ** 3 * disxi * disyi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 0] * img_fnodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 0] * disyi + img_fnodes[:, 1] * disxi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             1] * img_fnodes[:,
                                                                                  0] * Dx1 * H3ski / Ra ** 3) - A2 * Pyx
Mzx = MSzx - Ra ** 3 * disxi * diszi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 0] * img_fnodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 0] * diszi + img_fnodes[:, 2] * disxi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             2] * img_fnodes[:,
                                                                                  0] * Dx1 * H3ski / Ra ** 3) - A2 * Pzx
Mzy = MSzy - Ra ** 3 * disyi * diszi * H3ski / Xf ** 3 - A1 * (
        img_fnodes[:, 2] * img_fnodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
        img_fnodes[:, 2] * disyi + img_fnodes[:, 1] * diszi) / Xf ** 2 + 2 * img_fnodes[:,
                                                                             1] * img_fnodes[:,
                                                                                  2] * Dx1 * H3ski / Ra ** 3) - A2 * Pzy
