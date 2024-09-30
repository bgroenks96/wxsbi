#%%
import os.path as path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#%%
# rootdir = "/data/compoundx/causal_flood/basins_averaged"
# metadata_file = "basins_info.csv"
# metadata = pd.read_csv(path.join(rootdir, metadata_file))
# metadata

def data_var_name_map(time="time", prec="prec", Tair_mean="Tair_mean", Tair_min="Tair_min", Tair_max="Tair_max"):
    return {
        'time': time,
        'prec': prec,
        'Tair_mean': Tair_mean,
        'Tair_min': Tair_min,
        'Tair_max': Tair_max,
    }
    
def load_time_series_csv(filepath, col_name_map=data_var_name_map(), **kwargs):
    data = pd.read_csv(filepath, **kwargs)
    timedim = col_name_map["time"]
    data[timedim] = pd.to_datetime(data[timedim])
    data = data.set_index(timedim)
    return data.rename(columns={v: k for k,v in col_name_map.items()})
#%%
class EOBSBasinDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, datadir: str, summarizer=lambda x: x.mean(axis=0).reset_index(),
                 basin_features=["lat","lon","area","altitude_basin","forest","slope"],
                 data_vars=["tavg","tmax","tmin","pre"], **kwargs):
        super().__init__(**kwargs)
        self.datadir = datadir
        self.basin_features = basin_features
        self.data_vars = data_vars
        self.dataset = self._load_dataset(metadata)
        self.metadata = metadata.loc[metadata.id.isin(self.dataset.id.unique())]
        self.summary_stats = self._compute_summary_stats(summarizer)
        self.standardizer = self._compute_standardizer()

    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        id = row.id
        data = self.dataset.loc[self.dataset.id == id]
        feats = row[self.basin_features].astype(np.float64)
        stats = self.summary_stats.loc[self.summary_stats.id == id]
        return feats, data[["time"] + self.data_vars], stats.iloc[:,2:]
    
    def _load_dataset(self, metadata):
        basins = []
        for i in tqdm(range(len(metadata.index)), desc="loading basin data"):
            row = metadata.iloc[i]
            filepath = path.join(self.datadir, f"{row.id}.csv")
            if path.isfile(filepath) and np.all(np.isfinite(row[self.basin_features].astype(np.float64))):
                data = pd.read_csv(filepath)
                data["id"] = row.id
                basins.append(data[["id","time"] + self.data_vars])
        assert len(basins) > 0, f"no matching basin data files found at {self.datadir}"
        all_basins = pd.concat(basins, axis=0)
        all_basins["time"] = pd.to_datetime(all_basins["time"])
        return all_basins
    
    def _compute_standardizer(self):
        basin_features = self.metadata[self.basin_features]
        x_scale, x_shift = basin_features.mean(axis=0), basin_features.std(axis=0)
        y_scale, y_shift = self.dataset.iloc[:,2:].mean(axis=0), self.dataset.iloc[:,2:].std(axis=0)
        s_scale, s_shift = self.summary_stats.iloc[:,2:].mean(axis=0), self.summary_stats.iloc[:,2:].std(axis=0)
        feat_standardizer = {'scale': x_scale, 'shift': x_shift}
        data_standardizer = {'scale': y_scale, 'shift': y_shift}
        stats_standardizer = {'scale': s_scale, 'shift': s_shift}
        return {'features': feat_standardizer, 'data': data_standardizer, 'stats': stats_standardizer}
    
    def _compute_summary_stats(self, summarizer):
        summary_stats = self.dataset[self.data_vars] \
            .groupby([self.dataset.id, self.dataset.time.dt.year]) \
            .apply(summarizer) \
            .reset_index() \
            .drop(columns=["level_2"])
        return summary_stats
    
    def standardize(self, x, group="features"):
        shift = self.standardizer[group]["shift"]
        scale = self.standardizer[group]["scale"]
        return (x - shift) / (scale + 1e-8)
    
    def destandardize(self, x, group="features"):
        shift = self.standardizer[group]["shift"]
        scale = self.standardizer[group]["scale"]
        return x*(scale+1e-8) + shift
