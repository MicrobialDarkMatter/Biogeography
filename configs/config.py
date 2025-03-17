import os
import torch

from configs.data_folder_path import data_folder_path

config = {
    "data": {
        "X_path": os.path.join(data_folder_path, "full/X_pH.csv"),
        "Y_path": os.path.join(data_folder_path, "full/Y.csv"),
        "coords_path": os.path.join(data_folder_path, "full/XY.csv"),
        "traits_path": os.path.join(data_folder_path, "full/traits.csv"),
        "normalize_X": True,
    },
    "general": {
        "n_iter": 100,
        "n_particles": 8,
        "lr": 0.01,
        "batch_size": 512,
        "train_pct": 0.8,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "verbose": True,
    },
    "environmental": {
        "n_latents": 5,
        "n_inducing_points": 200,
    },
    "spatial": {
        "n_latents": 5,
        "n_inducing_points": 500,
    },
    "hmsc": {
        "k_folds": 5,
        "cross_validation": False,
        "likelihood": "bernoulli",
    },
    "additive": {  # To specify if certain components should be included or omitted.
        "environment": True,
        "spatial": True,
        "traits": True,
    }
}
