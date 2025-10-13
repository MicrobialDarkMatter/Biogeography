import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os

from EcoGP.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood

if __name__ == "__main__":
    from EcoGP.model import EcoGP
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from EcoGP.DataSampler import DataSampler
    from EcoGP.DataLoad import DataLoad

    from EcoGP.misc.calculate_metrics import calculate_metrics, precision_at_k

    from sklearn import metrics

    # # Add the parent directory (or any other directory where the config module is located) to the Python path
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

    from configs.config_toy import config  # Import the config module

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
    traits_path = config["data"]["traits_path"]
    total_counts_path = config["data"]["total_counts_path"]
    #hierarchy_path = config["data"]["hierarchy_path"]

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

    prevalence_threshold = config["data"]["prevalence_threshold"]

    save_model_path = config["general"]["save_model_path"]
    # STOP ARGUMENTS

    seed = 0  # TODO: Move to Config
    torch.manual_seed(seed)

    data = DataLoad(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=True,  # TODO: Move to Config
        total_counts_path=total_counts_path,
        presence_absence_Y=False,  # TODO: Move to Config
        verbose=False  # TODO: Move to Config
    )

    dataset = DataSampler(data)

    if spatial:
        train_indices, test_indices, validation_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                       split_pct,
                                                                       generator=torch.Generator().manual_seed(seed))

        # Getting the spatial locations split into separate sets
        train_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(train_indices.indices))]
        test_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(test_indices.indices))]
        validation_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(validation_indices.indices))]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
    else:
        train_dataset, test_dataset, validation_dataset = random_split(dataset, split_pct,
                                                                       generator=torch.Generator().manual_seed(seed))

    # # Make sure at least 10 species obserservations are present in each subset of the data
    # keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= 10) & (
    #             dataset.Y[test_dataset.indices].sum(dim=0) >= 10)
    # dataset.Y = dataset.Y[:, keep_y]
    # dataset.Y_cols_species = dataset.Y_cols_species[keep_y]
    # dataset.n_species = dataset.Y.shape[1]
    # if traits_path:
    #     dataset.traits = dataset.traits[keep_y, :]

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    # n_traits = dataset.n_traits
    unique_coordinates = dataset.unique_coords[
        dataset.get_dist_idx_reverse(train_dataset.indices)[0]] if spatial else None

    model = EcoGP(
        n_latents_env,
        n_variables,
        n_inducing_points_env,
        n_latents_spatial,
        n_inducing_points_spatial,
        unique_coordinates,
        environment=environment,
        spatial=spatial,
        traits=traits,
        likelihood=DirichletMultinomialLikelihood#BernoulliLikelihood#  # TODO: Move to Config
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    from EcoGP.BetaTraceELBO import BetaTraceELBO

    elbo = BetaTraceELBO(beta=.5, num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    losses = []

    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(batch) / batch.get("Y").nelement()

        iterator.set_postfix(loss=loss)
        losses.append(loss)

    plt.plot(list(range(n_iter)), losses)
    plt.show()

    # Save model
    if save_model_path:
        torch.save(model, os.path.join(save_model_path, "model.pt"))
        pyro.get_param_store().save(
            os.path.join(save_model_path, "param_store.pt"))  # f"../results/saved_models/param_store.pt"
        torch.save(dataset, os.path.join(save_model_path, "dataset.pt"))

    # Testing
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        batch = test_dataset.dataset.get_batch_data(idx)
        batch["training"] = False
        batch["do_spatial"] = False

        predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=25)

        y = predictive(batch)["y"].mean(dim=0).squeeze()

        prob_list.append(y)
        y_test_list.append(batch.get("Y") / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    test_Y = torch.concat(y_test_list)
    del prob_list, y_test_list

    torch.save(prob, os.path.join(save_model_path, "Y_pred_test.pt"))
    torch.save(test_Y, os.path.join(save_model_path, "Y_true_test.pt"))

    from EcoGP.misc.metrics import precision_at_k

    print(metrics.mean_absolute_error(test_Y, prob))
    print(precision_at_k(test_Y, prob, k=20).mean())

    print("Done")

    # Validation
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    prob_list = []
    y_validation_list = []
    for idx in validation_dataloader:
        batch = validation_dataset.dataset.get_batch_data(idx)
        batch["training"] = False
        batch["do_spatial"] = False

        predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=25)

        y = predictive(batch)["y"].mean(dim=0).squeeze()

        prob_list.append(y)
        y_validation_list.append(batch.get("Y") / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    validation_Y = torch.concat(y_validation_list)
    del prob_list, y_validation_list

    torch.save(prob, os.path.join(save_model_path, "Y_pred_validation.pt"))
    torch.save(validation_Y, os.path.join(save_model_path, "Y_true_validation.pt"))
