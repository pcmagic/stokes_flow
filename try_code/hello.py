sij_1tail = lambda theta, ph, rt1, rt2: np.array(((((
                                                            16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (2. * (
        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
        theta)) ** 1.5), 0, 0), (0, ((8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (1 + 3 * np.cos(2 * theta))) / (2. * (
        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
        theta)) ** 1.5), (ph * np.pi * rt1 * theta * np.sin(theta)) / (
                                         8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                         theta)) ** 1.5), (0, (
        ph * np.pi * rt1 * theta * np.sin(theta)) / (
                                                                   8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) ** 1.5, (
                                                                   4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) / (
                                                                   8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) ** 1.5)))
sij = lambda theta, ph, rt1, rt2: np.array(((((-(
        (16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta)) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (
                                                      8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                      theta)) ** 1.5 + ((
                                                                                16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (
                                                      8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                      theta)) ** 1.5) / 2., 0, 0), (0, ((-(
        (8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta)) - 2 * np.pi ** 2 * rt1 ** 2 * (1 + 3 * np.cos(2 * theta))) / (
                                                                                                8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                theta)) ** 1.5 + (
                                                                                                (
                                                                                                        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
                                                                                                theta) - 2 * np.pi ** 2 * rt1 ** 2 * (
                                                                                                        1 + 3 * np.cos(
                                                                                                        2 * theta))) / (
                                                                                                8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                theta)) ** 1.5) / 2.,
                                                                                    ph * np.pi * rt1 * theta * (
                                                                                            (
                                                                                                    8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                    theta)) ** (
                                                                                                -1.5) - (
                                                                                                    8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                    theta)) ** (
                                                                                                -1.5)) * np.sin(
                                                                                            theta)),
                                            (0,
                                             ph * np.pi * rt1 * theta * (
                                                     (
                                                             8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                             theta)) ** (
                                                         -1.5) - (
                                                             8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                             theta)) ** (
                                                         -1.5)) * np.sin(
                                                     theta),
                                             (
                                                     4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) / (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 1.5 + (
                                                     4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) / (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 1.5)))
dij = lambda theta, ph, rt1, rt2: np.array((((rt2 ** 2 * np.cos(theta) * (
        8 * np.pi ** 4 * rt1 ** 2 - 2 * ph ** 2 * np.pi ** 2 * theta ** 2 + 8 * np.pi ** 4 * rt1 ** 2 * np.cos(
        theta) * (-4 + 3 * np.cos(theta))) * (-1 + (
        4 * np.pi ** 2 * rt1 ** 2 * np.sin(theta) ** 2) / (
                                                      ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2))) / (
                                                     2. * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5), (
                                                     -48 * np.pi ** 6 * rt1 ** 4 * rt2 ** 2 * (
                                                     -1 + np.cos(theta)) * np.cos(
                                                     theta) ** 2 * np.sin(theta) ** 2) / (
                                                     (ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5), (
                                                     -12 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * (
                                                     -1 + np.cos(theta)) * np.sin(
                                                     theta)) / (
                                                     (ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5)), ((
                                                                                -48 * np.pi ** 6 * rt1 ** 4 * rt2 ** 2 * (
                                                                                -1 + np.cos(
                                                                                theta)) * np.cos(
                                                                                theta) ** 2 * np.sin(
                                                                                theta) ** 2) / (
                                                                                (
                                                                                        ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                        theta)) ** 2.5),
                                                                        -((
                                                                                  np.pi ** 2 * rt2 ** 2 * np.cos(
                                                                                  theta) * (
                                                                                          -1 + (
                                                                                          4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                          theta) ** 2) / (
                                                                                                  ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2)) * (
                                                                                          2 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 2 * np.pi ** 2 * rt1 ** 2 * (
                                                                                          -4 * np.cos(
                                                                                          theta) + 3 * np.cos(
                                                                                          2 * theta)))) / (
                                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                  theta)) ** 2.5),
                                                                        (
                                                                                6 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * np.sin(
                                                                                2 * theta)) / ((
                                                                                                       ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                                       8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                       theta)) ** 2.5)),
                                            ((
                                                     -12 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * (
                                                     -1 + np.cos(theta)) * np.cos(
                                                     theta) * np.sin(theta)) / (
                                                     (ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5), (
                                                     6 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * np.cos(
                                                     theta) * np.sin(2 * theta)) / (
                                                     (ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5), (
                                                     -8 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * (
                                                     -4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta))) / (
                                                     (ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                     theta)) ** 2.5))))
