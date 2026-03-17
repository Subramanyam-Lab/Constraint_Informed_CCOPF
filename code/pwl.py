import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

###################################################
### Use this script to find optimal pwl segments 
### when specifying a tolerance, default to 0.005
###################################################
def pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def error_formula(a: float, s: float) -> float:
    if s <= 0:
        return -1.0

    inside = -math.log(2.0 * math.pi * s * s)
    if inside <= 0:
        return sys.float_info.max

    x_star = math.sqrt(inside)
    return Phi(x_star) - s * (x_star - a) - Phi(a)

def segment_error(a: float, b: float) -> float:
    if b <= a:
        return -1.0
    num = Phi(b) - Phi(a)
    den = b - a
    if abs(den) < 1e-15:
        return sys.float_info.max
    s = num / den
    return error_formula(a, s)

def find_next_b(a: float, tol: float,
                left_guess: float = 1e-8,
                right_guess: float = 5.0,
                max_iter: int = 100,
                eps: float = 1e-12) -> float:

    def f(bval):
        return segment_error(a, bval) - tol

    left  = a + left_guess
    right = a + right_guess

    fL = f(left)
    fR = f(right)

    while fL * fR > 0 and right < 1e6:
        right *= 2.0
        fR = f(right)
        if right > 1e6:
            return None

    if fL * fR > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        fm  = f(mid)

        if abs(fm) < eps:
            return mid

        if fm * fL > 0:
            left = mid
            fL   = fm
        else:
            right = mid
            fR    = fm

    return 0.5*(left + right)

def build_segments(tol: float):
    t_points = [0.0]
    while True:
        m = len(t_points) - 1
        if Phi(t_points[m]) >= 1.0 - tol:
            break

        b_val = find_next_b(a = t_points[m], tol = tol)
        if b_val is None:
            print("Warning: cannot find next segment with error == tol. Stopping.")
            break

        t_points.append(b_val)

    return t_points

def piecewise_linear_phi_approx(x: float, t_points: list) -> float:
    if x <= t_points[0]:
        return 0.0

    if x >= t_points[-1]:
        return Phi(t_points[-1])

    for i in range(len(t_points) - 1):
        if t_points[i] <= x <= t_points[i+1]:
            denom = t_points[i+1] - t_points[i]
            if abs(denom) < 1e-15:
                return Phi(t_points[i])
            s = (Phi(t_points[i+1]) - Phi(t_points[i])) / denom
            return Phi(t_points[i]) + s * (x - t_points[i])

    return Phi(t_points[-1])

def plot_segments(t_points: list, tol: float):
    plt.close("all")
    
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

if __name__ == "__main__":
    TOL = 0.005

    nodes = build_segments(TOL)
    print("\n======================================")
    print("Segmentation results (automatic M):")
    print("======================================")
    print(f"Number of segments used = {len(nodes)-1}")
    for i, t in enumerate(nodes):
        print(f" t_{i} = {t:.8f},  Phi(t_{i}) = {Phi(t):.8f}")

    plot_segments(nodes, TOL)

as_list = []
bs_list = []

for i in range(len(nodes) - 1):
    t_i, t_ip1 = nodes[i], nodes[i+1]
    phi_i = Phi(t_i)
    phi_ip1 = Phi(t_ip1)

    slope = (phi_ip1 - phi_i) / (t_ip1 - t_i)
    intercept = phi_i - slope * t_i

    as_list.append(slope)
    bs_list.append(intercept)

as_list.append(0.0)
bs_list.append(Phi(nodes[-1]))

with open("pwl_segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["slope", "intercept"])
    for a, b in zip(as_list, bs_list):
        writer.writerow([a, b])

print("\n Saved piecewise segment slopes and intercepts to pwl_segments.csv")