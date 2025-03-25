#!/usr/bin/env python3
"""
normal_segmentation_slope_minimal_plot.py

We want to split the x-axis [0,∞) into segments [t_0, t_1], [t_1, t_2], ...
such that on each segment, the linear interpolation of the standard normal CDF
has an approximate maximum error = 'tol'.

We do *not* specify a fixed number of segments M in advance.
Instead, the algorithm automatically adds segments until:
    Phi(t_m) >= 1 - tol,
meaning the CDF is within tol of 1.

This script also produces a plot to visualize:
  - The exact standard normal CDF, Phi(x).
  - The piecewise-linear approximation built from the segments.
  - The segment boundary nodes.

Author: <Your Name/Date>
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------------------------------------------------
# 1. Basic functions: standard normal PDF and CDF
# -------------------------------------------------------------------------

def pdf(x: float) -> float:
    """
    Standard normal PDF: phi(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2).
    """
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def Phi(x: float) -> float:
    """
    Standard normal CDF, using math.erf.
    Phi(x) = 0.5 * [1 + erf(x / sqrt(2))]
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -------------------------------------------------------------------------
# 2. The error formula in terms of slope = [Phi(b) - Phi(a)] / (b-a).
# -------------------------------------------------------------------------

def error_formula(a: float, s: float) -> float:
    """
    E(a, s) = Phi( sqrt(-ln(2*pi*s^2)) )
              - s * ( sqrt(-ln(2*pi*s^2)) - a )
              - Phi(a).

    We assume s = [Phi(b) - Phi(a)] / (b-a).
    """
    if s <= 0:
        # Negative or zero slope => invalid
        return -1.0

    inside = -math.log(2.0 * math.pi * s * s)
    if inside <= 0:
        # sqrt(...) would be imaginary => invalid
        return sys.float_info.max

    x_star = math.sqrt(inside)
    return Phi(x_star) - s * (x_star - a) - Phi(a)

def segment_error(a: float, b: float) -> float:
    """
    Compute slope s = (Phi(b) - Phi(a)) / (b - a),
    then evaluate error_formula(a, s).
    This yields the error on [a,b].
    """
    if b <= a:
        return -1.0  # invalid domain
    num = Phi(b) - Phi(a)
    den = b - a
    if abs(den) < 1e-15:
        return sys.float_info.max
    s = num / den
    return error_formula(a, s)


# -------------------------------------------------------------------------
# 3. Bisection method to find b > a such that segment_error(a, b) = tol
# -------------------------------------------------------------------------

def find_next_b(a: float, tol: float,
                left_guess: float = 1e-8,
                right_guess: float = 5.0,
                max_iter: int = 100,
                eps: float = 1e-12) -> float:
    """
    Given the current endpoint 'a',
    find b > a such that segment_error(a, b) = tol, using bisection.

    1) We'll define f(b) = segment_error(a, b) - tol.
    2) We'll search in [a+left_guess, a+right_guess].
       If we can't find a sign change, we expand 'right' up to 1e6.

    Returns b if successful, or None if we fail to bracket or converge.
    """

    def f(bval):
        return segment_error(a, bval) - tol

    left  = a + left_guess
    right = a + right_guess

    fL = f(left)
    fR = f(right)

    # If not bracketed, try expanding 'right'
    while fL * fR > 0 and right < 1e6:
        right *= 2.0
        fR = f(right)
        if right > 1e6:
            # arbitrary cutoff
            return None

    # If still no sign change => no bracket
    if fL * fR > 0:
        return None

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        fm  = f(mid)

        if abs(fm) < eps:
            return mid

        if fm * fL > 0:
            # same sign => move left up
            left = mid
            fL   = fm
        else:
            right = mid
            fR    = fm

    return 0.5*(left + right)


# -------------------------------------------------------------------------
# 4. Main routine (no M). We automatically stop when Phi(t_m) >= 1 - tol.
# -------------------------------------------------------------------------

def build_segments(tol: float):
    """
    Build segments [0, t_1], [t_1, t_2], ... 
    until we reach Phi(t_m) >= 1 - tol.

    Return the list of t-values [t_0, t_1, ..., t_m].
    """
    t_points = [0.0]  # start at 0
    while True:
        m = len(t_points) - 1
        # If we've reached near 1 in the CDF, stop
        if Phi(t_points[m]) >= 1.0 - tol:
            break

        # Otherwise, find next b so that error ~ tol
        b_val = find_next_b(a = t_points[m], tol = tol)
        if b_val is None:
            print("Warning: cannot find next segment with error == tol. Stopping.")
            break

        t_points.append(b_val)

    return t_points


# -------------------------------------------------------------------------
# 5. To plot: piecewise-linear approximation
# -------------------------------------------------------------------------

def piecewise_linear_phi_approx(x: float, t_points: list) -> float:
    """
    Given the segment boundaries t_points and the standard normal CDF Phi,
    return the piecewise-linear approximation at x.

    The segments are [t_i, t_{i+1}].
    For x in [t_i, t_{i+1}],
      slope = (Phi(t_{i+1}) - Phi(t_i)) / (t_{i+1} - t_i)
      L(x) = Phi(t_i) + slope * (x - t_i)

    We do a simple linear scan for demonstration (O(#segments)).
    """
    # If x < t_points[0], approximate is 0
    if x <= t_points[0]:
        return 0.0

    # If x > t_points[-1], approximate is Phi(t_points[-1]) or saturate near 1
    if x >= t_points[-1]:
        return Phi(t_points[-1])

    # Otherwise, find the interval [t_i, t_{i+1}]
    for i in range(len(t_points) - 1):
        if t_points[i] <= x <= t_points[i+1]:
            # slope
            denom = t_points[i+1] - t_points[i]
            if abs(denom) < 1e-15:
                return Phi(t_points[i])
            s = (Phi(t_points[i+1]) - Phi(t_points[i])) / denom
            return Phi(t_points[i]) + s * (x - t_points[i])

    # Fallback: shouldn't reach here
    return Phi(t_points[-1])


def plot_segments(t_points: list, tol: float):

    x_min = t_points[0]
    x_max = t_points[-1]
    X = np.linspace(x_min, x_max, 500)

    Y_actual = [Phi(x) for x in X]
    Y_approx = [piecewise_linear_phi_approx(x, t_points) for x in X]

    plt.figure(figsize=(8,6))
    plt.plot(X, Y_actual, label='Exact Φ(x)', linewidth=2)
    plt.plot(X, Y_approx, label='PWL Approx', linestyle='--', linewidth=2)
    plt.scatter(t_points, [Phi(t) for t in t_points], color='red', zorder=5, label='Segment Nodes')

    plt.title(f"PWL Approximation of Standard Normal CDF (tol={tol})")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------
# 6. Demo usage
# -------------------------------------------------------------------------

if __name__ == "__main__":
    TOL = 0.005  # You can set this to any desired tolerance

    nodes = build_segments(TOL)
    print("\n======================================")
    print("Segmentation results (automatic M):")
    print("======================================")
    print(f"Number of segments used = {len(nodes)-1}")
    for i, t in enumerate(nodes):
        print(f" t_{i} = {t:.8f},  Phi(t_{i}) = {Phi(t):.8f}")

    # Show a plot
    plot_segments(nodes, TOL)

# Compute slopes and intercepts
as_list = []
bs_list = []

for i in range(len(nodes) - 1):  # From t_0 to t_5
    t_i, t_ip1 = nodes[i], nodes[i+1]
    phi_i = Phi(t_i)
    phi_ip1 = Phi(t_ip1)

    slope = (phi_ip1 - phi_i) / (t_ip1 - t_i)
    intercept = phi_i - slope * t_i

    as_list.append(slope)
    bs_list.append(intercept)

# Final flat segment [t_5, ∞)
as_list.append(0.0)
bs_list.append(Phi(nodes[-1]))

# Write to CSV
with open("pwl_segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["slope", "intercept"])
    for a, b in zip(as_list, bs_list):
        writer.writerow([a, b])

print("\n Saved piecewise segment slopes and intercepts to pwl_segments.csv")
