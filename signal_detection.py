from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt

class SignalDetection:
    def SignalDetection(hits, misses, false_alarms, correct_rejections):
        # chekcing for if the input are valid
        for value in [hits, misses, false_alarms, correct_rejections]:
            if not isinstance(value, int) or i < 0:
                raise ValueError("All input must be a non-negative integer.")
            if not np.isfinite(value):
                raise ValueError("All input must be a finite number.")
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        denom = self.hits + self.misses
        if denom == 0:
            raise ValueError("Cannot find hit rate because Hits and misses cannot both be 0.")
        return self.hits / denom

    def false_alarm_rate(self):
        denom = self.false_alarms + self.correct_rejections
        if denom == 0:
            raise ValueError("Cannot find false alarm rate because False alarms and correct rejections cannot both be 0.")
        return self.false_alarms / denom

    def d_prime(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()
        if h <= 0 or h >= 1 or fa <= 0 or fa >= 1:
            raise ValueError("d' is not finite when hit rate or false alarm rate is 0 or 1.")
        return norm.ppf(h) - norm.ppf(fa)

    def criterion(self):
        h = self.hit_rate()
        fa = self.false_alarm_rate()
        if h <= 0 or h >= 1 or fa <= 0 or fa >= 1:
            raise ValueError("Criterion is not finite when hit rate or false alarm rate is 0 or 1.")
        return -0.5 * (norm.ppf(h) + norm.ppf(fa))

    def __add__(self, other):
        if not isinstance(other, SignalDetection):
            raise TypeError("Can only add SignalDetection object to SignalDetection object, got {type(other).__name__}")
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )

    def __sub__(self, other):
       if not isinstance(other, SignalDetection):
            raise TypeError("Can only add SignalDetection object to SignalDetection object, got {type(other).__name__}")
        return SignalDetection(
            self.hits - other.hits,
            self.misses - other.misses,
            self.false_alarms - other.false_alarms,
            self.correct_rejections - other.correct_rejections
        )

    def __mul__(self, factor):
        if not isinstance(factor, int) or factor < 0:
            raise ValueError("factor must be a non-negative integer.")
        if not np.isfinite(factor):
            raise ValueError("factor must be finite.")
        if factor == 0:
            logger.info("SignalDetection multiplied by 0; all counts become 0.")
        return SignalDetection(
            self.hits * factor,
            self.misses * factor,
            self.false_alarms * factor,
            self.correct_rejections * factor,
        )

    @staticmethod
    def plot_roc(sdt_list):
        # Validate input
        if not all(isinstance(sdt, SignalDetection) for sdt in sdt_list):
            raise TypeError("All elements must be SignalDetection objects")

        xs = [0.0]
        ys = [0.0]

        for sdt in sdt_list:
            xs.append(float(sdt.hit_rate()))
            ys.append(float(sdt.false_alarm_rate()))

        xs.append(1.0)
        ys.append(1.0)

        points = sorted(zip(xs, ys), key=lambda p: p[0])
        xs, ys = zip(*points)

        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Hit rate")
        ax.set_ylabel("False alarm rate")
        return fig, ax