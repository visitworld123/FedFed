import logging
from copy import deepcopy

class MaxMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.max = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max


class MinMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.min = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.min is None or value < self.min:
            self.min = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.min

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        sum_key = key + "/" + "sum"
        count_key = key + "/" + "count"
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        min_key = key + "/" + "min"
        final_key = key + "/" + "final"
        summary = {
            sum_key: self.sum,
            count_key: self.count,
            avg_key: self.avg,
            max_key: self.max,
            min_key: self.min,
            final_key: self.val,
        }
        return summary









