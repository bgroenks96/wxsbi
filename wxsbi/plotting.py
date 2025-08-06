import matplotlib.pyplot as plt

from .wxsbi import SBIResults


def plot_target_densitites_sbi(results: SBIResults, cmap="Dark2"):
    summary_target = results.summary_target
    summary_names = results.simulator.summarizer.names
    fig, axs = plt.subplots(1, summary_target.shape[0], figsize=(summary_target.shape[0] * 5, 5))
    cmap = plt.cm.get_cmap(cmap).colors
    if summary_target.shape[0] == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        handles = []
        if "calibration_posterior" in results.simulations:
            _, _, h1 = axs[i].hist(
                results.simulations["calibration_posterior"][:, i],
                color=cmap[0],
                bins=20,
                density=True,
                alpha=0.5,
                label="Calibration posterior",
            )
        if "calibration_posterior_mean" in results.simulations:
            _, _, h2 = axs[i].hist(
                results.simulations["calibration_posterior_mean"][:, i],
                color=cmap[1],
                bins=20,
                density=True,
                alpha=0.5,
                label="Calibration posterior mean",
            )
        if "sbi_prior" in results.simulations:
            _, _, h3 = axs[i].hist(
                results.simulations["sbi_prior"][:, i],
                color="gray",
                bins=20,
                density=True,
                alpha=0.5,
                label="SBI prior",
            )
        if "sbi_posterior" in results.simulations:
            _, _, h4 = axs[i].hist(
                results.simulations["sbi_posterior"][:, i],
                color=cmap[2],
                bins=20,
                density=True,
                alpha=0.5,
                label="SBI posterior",
            )
        if "sbi_posterior_map" in results.simulations:
            _, _, h5 = axs[i].hist(
                results.simulations["sbi_posterior_map"][:, i],
                color=cmap[3],
                bins=20,
                density=True,
                alpha=0.5,
                label="SBI posterior MAP",
            )
        axs[i].set_title(summary_names[i])
        target_line = axs[i].axvline(summary_target[i].flatten(), color="black", label="Target")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=6)
    return fig, axs
