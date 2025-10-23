
from configs.config_clean_unique import config  # Microflora Danica

import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import tqdm

import gpytorch

from torch.utils.data import DataLoader, random_split
from ExploreSpatial.DataSampler import DataSampler
from ExploreSpatial.DataLoad import DataLoad

from sklearn import metrics


def calculate_metrics(y_true, y_pred):
    auc_per_species = [
        metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    auc = torch.tensor(auc_per_species)
    auc = (auc[~torch.isnan(auc)]).mean().item()
    return auc


# if __name__ == "__main__":

# ARGUMENTS
environment = config["additive"]["environment"]
spatial = config["additive"]["spatial"]
traits = config["additive"]["traits"]

x_path = config["data"]["X_path"]  # .replace("clean", "full")
y_path = config["data"]["Y_path"]  # .replace("clean", "full")
coords_path = config["data"]["coords_path"]  # .replace("clean", "full")
traits_path = config["data"]["traits_path"]
total_counts_path = config["data"]["total_counts_path"]
# hierarchy_path = config["data"]["hierarchy_path"]

n_latents_env = config["environmental"]["n_latents"]
n_latents_spatial = config["spatial"]["n_latents"]
n_iter = config["general"]["n_iter"]
n_particles = config["general"]["n_particles"]
device = config["general"]["device"]
lr = config["general"]["lr"]
batch_size = config["general"]["batch_size"]
split_pct = config["general"]["split_pct"]
n_inducing_points_env = config["environmental"]["n_inducing_points"]
n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

verbose = config["general"]["verbose"]
presence_absence = config["data"]["presence_absence"]
normalize_X = config["data"]["normalize_X"]
likelihood = config["general"]["likelihood"]
seed = config["general"]["seed"]

# prevalence_threshold = config["data"]["prevalence_threshold"]

save_model_path = config["general"]["save_model_path"]
# STOP ARGUMENTS

torch.manual_seed(seed)

data = DataLoad(
    Y_path=y_path,
    X_path=x_path,
    coords_path=coords_path,
    traits_path=traits_path,
    device=device,
    normalize_X=normalize_X,
    total_counts_path=total_counts_path,
    presence_absence_Y=presence_absence,
    verbose=verbose
)

dataset = DataSampler(data)

# Split Data into sets
train_dataset, test_dataset, validation_dataset = random_split(dataset, split_pct,
                                                               generator=torch.Generator().manual_seed(seed))

# Make sure at least 1 species obserservations are present all splits
# Can't make predictions for a species not present in training
keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & \
         (dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & \
         (dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
dataset.Y = dataset.Y[:, keep_y]
dataset.taxon_names = dataset.taxon_names[keep_y]
dataset.n_species = dataset.Y.shape[1]

# Dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# TODO: Input as coordinates or environmental features!
feature_input = "coords"
print(f"Input data corresponds to {feature_input=}")

Y = dataset.Y[train_dataset.indices]
I, J = Y.shape
num_latents = n_latents_spatial
num_tasks = dataset.n_species

if feature_input == "coords":
    coords = dataset.coords[train_dataset.indices]
    n_S = coords.shape[1]
    # Creating a meshgrid of inducing points with min and max observed values
    a = torch.linspace(coords[:, 0].min(), coords[:, 0].max(), 4)
    b = torch.linspace(coords[:, 1].min(), coords[:, 1].max(), 4)
    A, B = torch.meshgrid(a, b, indexing="ij")
    inducing_points = torch.stack([A.flatten(), B.flatten()], dim=1).repeat(num_latents, 1, 1)

    features = coords

    test_x = dataset.coords[test_dataset.indices]
    test_y = dataset.Y[test_dataset.indices]

elif feature_input == "X":
    X = dataset.X[train_dataset.indices]
    n_S = X.shape[1]

    inducing_points = torch.randn(num_latents, 16, n_S)

    features = X

    test_x = dataset.X[test_dataset.indices]
    test_y = dataset.Y[test_dataset.indices]

else:
    print("WARNING: Incorrect feature input!!!")

