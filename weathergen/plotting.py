import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from functools import partial


# QQplots
def build_qq_plot(obs, sim, ax=None):
    obs_sorted = np.sort(obs.squeeze())
    sim_sorted = np.sort(sim.squeeze(), axis=1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(
        obs_sorted,
        y=np.median(sim_sorted, axis=0),
        yerr=np.stack(
            [
                np.median(sim_sorted, axis=0) - np.quantile(sim_sorted, 0.1, axis=0),
                np.quantile(sim_sorted, 0.9, axis=0) - np.median(sim_sorted, axis=0),
            ]
        ),
        marker="o",
    )
    ax.axline((0, 0), slope=1, c="black")
    ax.set_ylabel("Simulated")
    ax.set_xlabel("Observed")


def build_individual_qq_plot(obs, sim, ax=None):
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(sim)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x=obs_sorted, y=sim_sorted)
    ax.axline((0, 0), slope=1, c="black")


def build_ts_plot(obs, sim, time, i=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(time, obs.squeeze(), label="Observed")
    ax.plot(time, sim.squeeze()[i, :], label="Simulated", alpha=0.5)
    ax.legend()


def build_qq_plot_by_season(obs, sim, time, figsize=(20, 5)):
    fig, axs = plt.subplots(1, 4, figsize=figsize)
    build_qq_plot(obs.squeeze()[time.month.isin([3, 4, 5])], sim.squeeze()[:, time.month.isin([3, 4, 5])], ax=axs[0])
    axs[0].set_title("Spring")
    build_qq_plot(obs.squeeze()[time.month.isin([6, 7, 8])], sim.squeeze()[:, time.month.isin([6, 7, 8])], ax=axs[1])
    axs[1].set_title("Summer")
    build_qq_plot(
        obs.squeeze()[time.month.isin([9, 10, 11])], sim.squeeze()[:, time.month.isin([9, 10, 11])], ax=axs[2]
    )
    axs[2].set_title("Autumn")
    build_qq_plot(obs.squeeze()[time.month.isin([12, 1, 2])], sim.squeeze()[:, time.month.isin([12, 1, 2])], ax=axs[3])
    axs[3].set_title("Winter")


# Overview plots


def ci_sims(x, level=0.9):
    lower_bound, upper_bound = 0.5 - level / 2, 0.5 + level / 2
    return (np.quantile(x, lower_bound), np.quantile(x, upper_bound))


def get_mean_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = df_preds.groupby([df_preds.time.dt.month, df_preds.variable]).mean().drop(columns="time").reset_index()

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].mean().reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Mean")


def get_std_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = df_preds.groupby([df_preds.time.dt.month, df_preds.variable]).std().drop(columns="time").reset_index()

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].std().reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Std Dev")


def get_max_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = df_preds.groupby([df_preds.time.dt.month, df_preds.variable]).max().drop(columns="time").reset_index()

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].max().reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Max")


def get_min_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = df_preds.groupby([df_preds.time.dt.month, df_preds.variable]).min().drop(columns="time").reset_index()

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].min().reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Min")


def get_acf_plot(obs, pred, time, lag=1, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = (
        df_preds.groupby([df_preds.time.dt.month, df_preds.variable])["pred"]
        .apply(lambda x: np.corrcoef(x[:(-lag)], x[lag:])[0, 1])
        .reset_index()
    )

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = (
        df_obs.groupby(df_obs.time.dt.month)["obs"]
        .apply(lambda x: np.corrcoef(x[:(-lag)], x[lag:])[0, 1])
        .reset_index()
    )

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title(f"ACF {lag}")


def get_cond_mean_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = (
        df_preds.groupby([df_preds.time.dt.month, df_preds.variable])["pred"]
        .apply(lambda x: x[x > 0].mean())
        .reset_index()
    )

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].apply(lambda x: x[x > 0].mean()).reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Cond mean")


def get_cond_std_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = (
        df_preds.groupby([df_preds.time.dt.month, df_preds.variable])["pred"]
        .apply(lambda x: x[x > 0].std())
        .drop(columns="time")
        .reset_index()
    )

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].apply(lambda x: x[x > 0].std()).reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Cond std dev")


def get_prop_wet_plot(obs, pred, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="pred")
    df_preds = (
        df_preds.groupby([df_preds.time.dt.month, df_preds.variable])["pred"]
        .apply(lambda x: np.mean(x > 0))
        .reset_index()
    )

    df_obs = pd.DataFrame({"obs": obs.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)["obs"].apply(lambda x: np.mean(x > 0)).reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="pred", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    sns.lineplot(df_obs, x="time", y="obs", ax=ax, color="black")
    ax.set_title("Prop wet")


def get_temp_precip_cor_plot(obs_temp, pred_temp, obs_precip, pred_precip, time, ax=None, levels=[0.9, 0.8, 0.7, 0.5]):
    if ax is None:
        fig, ax = plt.subplots()

    df_preds = pd.DataFrame(pred_temp.squeeze().T)
    df_preds["time"] = time
    df_preds = df_preds.melt(id_vars="time", value_name="temp")

    df_preds_precip = pd.DataFrame(pred_precip.squeeze().T)
    df_preds_precip["time"] = time
    df_preds_precip = df_preds_precip.melt(id_vars="time", value_name="precip")

    df_preds = df_preds.merge(df_preds_precip, on=["time", "variable"])

    df_preds = (
        df_preds.groupby([df_preds.time.dt.month, df_preds.variable])[["temp", "precip"]]
        .corr()
        .iloc[0::2, -1]
        .reset_index()
    )

    df_obs = pd.DataFrame({"temp": obs_temp.squeeze(), "precip": obs_precip.squeeze(), "time": time})
    df_obs = df_obs.groupby(df_obs.time.dt.month)[["temp", "precip"]].corr().iloc[0::2, -1].reset_index()

    for level in levels:
        sns.lineplot(
            df_preds, x="time", y="precip", ax=ax, linestyle="", errorbar=partial(ci_sims, level=level), legend=False
        )
    sns.lineplot(df_obs, x="time", y="precip", ax=ax, color="black")
    sns.lineplot(df_obs, x="time", y="precip", ax=ax, color="black")
    ax.set_title("Correlation temperature - precip")


# Grid
# fig, axs = plt.subplots(10, 10, figsize = (25, 25))
# for i, ax in enumerate(axs.ravel()):
#    build_qq_plot(basin_data_train["prec"].squeeze(), svi_preds["prec"].squeeze()[i, :], ax = ax)

"""
fig, axs = plt.subplots(2, 4, figsize = (20, 10))
get_mean_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[0, 0])
get_std_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[0, 1])
get_max_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[0, 2])
get_min_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[0, 3])
get_acf_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[1, 0], lag = 1)
get_acf_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[1, 1], lag = 2)
get_acf_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[1, 2], lag = 3)

fig, axs = plt.subplots(2, 5, figsize = (25, 10))
get_mean_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[0, 0])
get_std_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[0, 1])
get_max_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[0, 2])
get_prop_wet_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[0, 3])
get_cond_mean_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[0, 4])
get_cond_std_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[1, 0])
get_acf_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[1, 1], lag = 1)
get_acf_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[1, 2], lag = 2)
get_acf_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[1, 3], lag = 3)

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("pr", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(obs["prec"], svi_preds["prec"], ax = axs[0])
build_ts_plot(obs["prec"], svi_preds["prec"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(obs["prec"], svi_preds["prec"], basin_data_train.index, figsize = (20, 2.5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("Tavg", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(obs["Tavg"], svi_preds["Tavg"], ax = axs[0])
build_ts_plot(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(obs["Tavg"], svi_preds["Tavg"], basin_data_train.index, figsize = (20, 2.5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("Tmin", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(basin_data_train["Tair_min"], svi_preds["Tmin"], ax = axs[0])
build_ts_plot(basin_data_train["Tair_min"], svi_preds["Tmin"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(basin_data_train["Tair_min"], svi_preds["Tmin"], basin_data_train.index, figsize = (20, 2.5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("Tmax", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(basin_data_train["Tair_max"], svi_preds["Tmax"], ax = axs[0])
build_ts_plot(basin_data_train["Tair_max"], svi_preds["Tmax"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(basin_data_train["Tair_max"], svi_preds["Tmax"], basin_data_train.index, figsize = (20, 2.5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("Trange", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(obs["Trange"], svi_preds["Trange"], ax = axs[0])
build_ts_plot(obs["Trange"], svi_preds["Trange"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(obs["Trange"], svi_preds["Trange"], basin_data_train.index, figsize = (20, 2.5))
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 2.5))
fig.suptitle("Tskew", fontsize = 20)
plt.subplots_adjust(top=0.7)
build_qq_plot(obs["Tskew"], svi_preds["Tskew"], ax = axs[0])
build_ts_plot(obs["Tskew"], svi_preds["Tskew"], basin_data_train.index, ax = axs[1])
build_qq_plot_by_season(obs["Tskew"], svi_preds["Tskew"], basin_data_train.index, figsize = (20, 2.5))

"""
