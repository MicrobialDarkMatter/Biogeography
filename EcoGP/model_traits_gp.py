import torch
import pyro
import pyro.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os

from EcoGP.MultitaskVariationalStrategy import MultitaskVariationalStrategy
from EcoGP.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood


class EcoGP(pyro.nn.PyroModule):
    def __init__(self,
                 n_latents_env=None,
                 n_variables=None,
                 n_inducing_points_env=None,
                 n_latents_spatial=None,
                 n_inducing_points_spatial=None,
                 unique_coordinates=None,
                 environment=True,
                 spatial=True,
                 traits=True,
                 likelihood=None,
                 n_traits=None):
        super().__init__()

        self.likelihood = likelihood

        self.environment = environment
        self.spatial = spatial
        self.traits = traits

        assert self.environment + self.spatial + self.traits, f"Model cannot run without any components! {self.environment=}, {self.spatial =}, {self.traits=}"
        print(f"Running with components: {self.environment=}, {self.spatial=}, {self.traits=}")

        if self.environment:
            self.n_latents_env = n_latents_env
            self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables,
                                   n_inducing_points=n_inducing_points_env)

        if self.spatial:
            self.n_latents_spatial = n_latents_spatial
            self.g = SpatialGP(n_latents=n_latents_spatial, unique_coordinates=unique_coordinates,
                               n_inducing_points=n_inducing_points_spatial)

        if self.traits:
            self.n_latents_env = n_latents_env
            # Using same GP formulation as for the Environment
            self.t = EnvironmentGP(n_latents=n_latents_env, n_variables=n_traits,
                                   n_inducing_points=n_inducing_points_env)

    def model(self, batch):
        pyro.module("model", self)

        n_samples = batch.get("n_samples_batch")
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        z = 0

        if self.environment:
            f_dist = self.f.pyro_model(batch.get("X"), name_prefix="f_GP")

            # Use a plate here to mark conditional independencies
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            # if self.traits:
            #     gamma = pyro.param("gamma", torch.zeros(self.n_latents_env, n_traits))
            #     w_loc = batch.get("traits") @ gamma.T
            # else:
            #     w_loc = torch.zeros(n_species, self.n_latents_env)
            #
            # with species_plate:
            #     w = pyro.sample("w", dist.Normal(loc=w_loc, scale=torch.ones_like(w_loc)).to_event(1))
            # z = z + f_samples @ w.squeeze().reshape(n_species, self.n_latents_env).T

            w_dist = self.t.pyro_model(batch.get("traits"), name_prefix="t_GP")

            # Use a plate here to mark conditional independencies
            with pyro.plate("T_plate", dim=-1):
                # Sample from latent function distribution
                w_samples = pyro.sample(".t(traits)", w_dist)

            w_samples = w_samples if w_samples.shape == torch.Size([n_species, self.n_latents_env]) else w_samples.mean(
                dim=0).reshape(n_species, self.n_latents_env)

            z = z + f_samples @ w_samples.T

        if self.spatial:
            g_dist = self.g.pyro_model(batch.get("coords"), name_prefix="g_GP")

            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            g_samples = g_samples if g_samples.shape == torch.Size(
                [n_samples, self.n_latents_spatial]) else g_samples.mean(dim=0).reshape(
                n_samples, self.n_latents_spatial)
            # g_samples = g_samples if g_samples.shape == torch.Size(
            #     [batch["n_locs_batch"], self.n_latents_spatial]) else g_samples.mean(dim=0).reshape(
            #     batch["n_locs_batch"], self.n_latents_spatial)
            # g_samples = g_samples[batch["batch_inverse"]]

            # v = pyro.param("v", torch.randn(self.n_latents_spatial, n_species))
            v_loc = torch.zeros(self.n_latents_spatial, n_species)
            v_scale = torch.ones(self.n_latents_spatial, n_species)
            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

            z = z + g_samples @ v

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=torch.zeros(n_species), scale=torch.ones(n_species)))

        z = z + bias

        self.likelihood(z, batch, samples_plate, species_plate)

    def guide(self, batch):
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        if self.environment:
            # w_loc = pyro.param(
            #     "w_loc",
            #     torch.zeros(n_species, self.n_latents_env)
            # )
            #
            # # Shape: [n_species, n_latents_env, n_latents_env]
            # w_scale_tril = pyro.param(
            #     "w_scale_tril",
            #     0.1 * torch.eye(self.n_latents_env)
            #     .expand(n_species, self.n_latents_env, self.n_latents_env)
            #     .clone(),
            #     constraint=dist.constraints.lower_cholesky
            # )
            #
            # # -- CRITICAL PART: set dim=-1 so that species is the RIGHTMOST dimension.
            # with species_plate:
            #     # By default, MultivariateNormal(...):
            #     #   - batch shape = [n_species]
            #     #   - event shape = [n_latents_env]
            #     #
            #     # Placing the plate at dim=-1 forces the "event dimension" to be -2,
            #     # so physically the sample comes out [n_latents_env, n_species].
            #     w = pyro.sample(
            #         "w",
            #         dist.MultivariateNormal(w_loc, scale_tril=w_scale_tril)
            #     )

            w_dist = self.t.pyro_guide(batch.get("traits"), name_prefix="t_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate("T_plate", dim=-1):
                # Sample from latent function distribution
                w_samples = pyro.sample(".t(traits)", w_dist)


            # pyro.module(self.name_prefixes[i], self.gp_models[i])
            f_dist = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

        if self.spatial:
            g_dist = self.g.pyro_guide(batch.get("coords"), name_prefix="g_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            v_loc = pyro.param("v_loc", torch.zeros(self.n_latents_spatial, n_species))
            v_scale = pyro.param(
                "v_scale",
                0.1 * torch.ones(self.n_latents_spatial, n_species),
                constraint=dist.constraints.positive
            )

            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

        # if self.traits:
        #     bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        #     bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)
        #
        #     with species_plate:
        #         bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

        bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

    def forward(self, batch):
        # Point prediction
        z = 0

        if self.environment:
            f_samples = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP").mean
            w_samples = self.t.pyro_guide(batch.get("traits"), name_prefix="t_GP").mean

            z = z + f_samples @ w_samples.T

        if self.spatial:
            g_samples = self.g.pyro_guide(batch.get("coords"), name_prefix="g_GP").mean
            v = pyro.param("v_loc")

            z = z + g_samples @ v

        bias = pyro.param("bias_loc")

        z = z + bias

        if isinstance(self.likelihood, type(BernoulliLikelihood)):
            return dist.Bernoulli(logits=z).mean

        if isinstance(self.likelihood, type(DirichletMultinomialLikelihood)):
            return dist.Dirichlet(concentration=z).mean


class EnvironmentGP(gpytorch.models.ApproximateGP):
    def __init__(self, n_latents, n_variables, n_inducing_points):
        self.n_latents = n_latents
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.randn(n_latents, n_inducing_points, n_variables)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(
            lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
            batch_shape=torch.Size([n_latents]),
            ard_num_dims=n_variables,
        )

        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, n_variables)
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HaversineRBFKernel(gpytorch.kernels.Kernel):
    """A GPyTorch kernel that computes the Haversine distance and applies an RBF transformation."""

    has_lengthscale = True  # Allows GPyTorch to learn the lengthscale

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """Compute the kernel matrix using Haversine distance with RBF transformation."""
        if diag:
            return torch.ones(1, x1.shape[-2])
        # Convert degrees to radians
        RADIUS = 6373  # Approximate radius of Earth in km

        # Convert degrees to radians
        lon1, lat1, lon2, lat2 = map(torch.deg2rad, (x1[:, :, 0], x1[:, :, 1], x2[:, :, 0], x2[:, :, 1]))

        # Compute differences
        dlon = lon2.unsqueeze(1) - lon1.unsqueeze(2)
        dlat = lat2.unsqueeze(1) - lat1.unsqueeze(2)

        # Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1.unsqueeze(2)) * torch.cos(lat2.unsqueeze(1)) * torch.sin(
            dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        haversine_dist = RADIUS * c

        # Apply the RBF kernel
        rbf_kernel = torch.exp(-0.5 * (haversine_dist / self.lengthscale) ** 2)

        return rbf_kernel


class SpatialGP(gpytorch.models.ApproximateGP):
    def __init__(self, n_latents, unique_coordinates, n_inducing_points):
        self.n_latents = n_latents
        num_coords = unique_coordinates.size(0)

        inducing_points = unique_coordinates[
                          torch.stack([torch.randperm(num_coords)[:n_inducing_points] for _ in range(self.n_latents)]),
                          :]

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(
            # HaversineRBFKernel(  # gpytorch.kernels.RBFKernel(#HaversineRBFKernel(  # CustomSpatialKernel(#
            lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
            batch_shape=torch.Size([n_latents]),
        )
        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, 1) * 5
        # self.covar_module.base_kernel.lengthscale = torch.ones(n_latents, 1, 1, requires_grad=False) * 3
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    import torch
    import pyro
    import pyro.distributions as dist
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import tqdm

    import wandb
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from torch.utils.data import DataLoader, random_split
    from EcoGP.DataSampler import DataSampler
    from EcoGP.DataLoad import DataLoad
    from EcoGP.BetaTraceELBO import BetaTraceELBO

    from configs.config_traits import config

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
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

    # Make sure at least 1 species obserservations are present all splits
    # Can't make predictions for a species not present in training
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
            dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                     dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
    dataset.Y = dataset.Y[:, keep_y]
    if dataset.using_total_counts:
        dataset.total_counts = (
                    (dataset.Y / dataset.total_counts).sum(dim=1) * dataset.total_counts.squeeze()).int().reshape(-1, 1)
    dataset.taxon_names = dataset.taxon_names[keep_y]
    dataset.n_species = dataset.Y.shape[1]
    if traits_path:
        dataset.traits = dataset.traits[keep_y, :]
    if verbose:
        print(f"Keeping {keep_y.sum().item()} taxons with at least {split_pct} * 10 "
              f"observations per split, respectively.")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    n_traits = dataset.n_traits
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
        likelihood=likelihood,
        n_traits=n_traits
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

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

    # # Save model
    # if save_model_path:
    #     torch.save(model, os.path.join(save_model_path, "model.pt"))
    #     pyro.get_param_store().save(os.path.join(save_model_path, "param_store.pt"))
    #     # torch.save(dataset, os.path.join(save_model_path, "dataset.pt"))
    #
    #     # Save config
    #     import pprint
    #
    #     with open(os.path.join(save_model_path, 'config.txt'), 'w') as f:
    #         # Create a PrettyPrinter object that writes to the file
    #         pp = pprint.PrettyPrinter(stream=f)
    #         pp.pprint(config)

    # Testing
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        batch = test_dataset.dataset.get_batch_data(idx)
        res = model.forward(batch).detach()

        prob_list.append(res)
        y_test_list.append(batch.get("Y") / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    test_Y = torch.concat(y_test_list)
    del prob_list, y_test_list

    # torch.save(prob, os.path.join(save_model_path, "Y_pred_test.pt"))
    # torch.save(test_Y, os.path.join(save_model_path, "Y_true_test.pt"))

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    metrics = calculate_metrics(test_Y, prob)
    print(metrics)