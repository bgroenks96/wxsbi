import matplotlib.pyplot as plt

from .wgen_sbi import SBIResults

def plot_target_densitites_sbi(results: SBIResults, cmap="Dark2"):
    obs_target = results.obs_target
    fig, axs = plt.subplots(1, obs_target.shape[0], figsize = (obs_target.shape[0]*5, 5))
    cmap = plt.cm.get_cmap(cmap).colors
    for i in range(obs_target.shape[0]):
        handles = []
        if "calibration_posterior" in results.simulations:
            _, _, h1 = axs[i].hist(results.simulations["calibration_posterior"][:, i], color=cmap[0], bins=20, density=True, alpha=0.5)
            handles.append(h1)
        if "calibration_posterior_mean" in results.simulations:
            _, _, h2 = axs[i].hist(results.simulations["calibration_posterior_mean"][:, i], color=cmap[1], bins=20, density=True, alpha=0.5)
            handles.append(h2)
        if "sbi_prior" in results.simulations:
            _, _, h3 = axs[i].hist(results.simulations["sbi_prior"][:, i], color="gray", bins=30, density=True, alpha=0.5)
            handles.append(h3)
        if "sbi_posterior" in results.simulations:
            _, _, h4 = axs[i].hist(results.simulations["sbi_posterior"][:, i], color=cmap[2], bins=30, density=True, alpha=0.5)
            handles.append(h4)
        if "sbi_posterior_map" in results.simulations:
            _, _, h5 = axs[i].hist(results.simulations["sbi_posterior_map"][:, i], color=cmap[3], bins=30, density=True, alpha=0.5)
            handles.append(h5)
        target_line = axs[i].axvline(obs_target[i].flatten(), color= "black", label="Target")
        handles.append(target_line)
    labels = ["Calibration", "Calibration (mean)", "SBI Prior", "SBI posterior", "SBI posterior (MAP)"]
    fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=6)
    return fig, axs
