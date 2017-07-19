_T_frame_left_hand = lambda self, R, B, s: np.vstack(
        (R * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
         (-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
         B * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T
_N_frame_left_hand = lambda self, R, B, s: np.vstack(
        ((-1) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
         (-1) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
         np.zeros_like(s))).T
_B_frame_left_hand = lambda self, R, B, s: np.vstack(
        (B * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
         (-1) * B * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
         (-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T
