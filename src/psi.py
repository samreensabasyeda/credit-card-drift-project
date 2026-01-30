import numpy as np
import pandas as pd

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_perc = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi
