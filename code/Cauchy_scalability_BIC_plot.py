import numpy as np
import matplotlib.pyplot as plt

###################################################
### Use this script to load BIC values and plot Fig.5
###################################################
results = np.load("../data/bic_results_all_methods.npy", allow_pickle=True).item()

def plot_all_curves(results):

    K = np.arange(1, len(results["classical_10"]) + 1)

    plt.figure(figsize=(8,6))

    plt.plot(
        K,
        results["classical_10"],
        color="royalblue",
        linestyle="--",
        marker="o",
        markersize=9,
        linewidth=3,
        label=r"Classical $|\mathcal{B}|=10$"
    )

    plt.plot(
        K,
        results["ci_regular_10"],
        color="darkorange",
        linestyle="--",
        marker="o",
        markersize=9,
        linewidth=3,
        label=r"Constraint-Informed $|\mathcal{B}|=10$"
    )

    classical50 = np.array(results["classical_50"])

    plt.plot(
        K,
        classical50,
        color="royalblue",
        linestyle="-",
        marker="*",
        markersize=12,
        linewidth=3,
        label=r"Classical $|\mathcal{B}|=50$"
    )

    plt.plot(
        K,
        results["ci_regular_50"],
        color="darkorange",
        linestyle="-",
        marker="*",
        markersize=12,
        linewidth=3,
        label=r"Constraint-Informed $|\mathcal{B}|=50$"
    )

    plt.draw()

    ymin, ymax = plt.gca().get_ylim()
    plt.plot(
        K[:2],
        [120000,120000],
        color="royalblue",
        linestyle="-.",
        marker="*",
        markersize=12,
        linewidth=3
    )

    # -----------------------------
    # Connect K=2 to K=3 manually
    # -----------------------------
    plt.plot(
        [K[1], K[2]],
        [120000, classical50[2]],
        color="royalblue",
        linestyle="-.",
        linewidth=3
    )
    plt.ylim(0,120000)
    plt.xlabel(r"$K$", fontsize=16)
    plt.ylabel("Best BIC Score", fontsize=16)
    plt.xticks(K, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()


plot_all_curves(results)