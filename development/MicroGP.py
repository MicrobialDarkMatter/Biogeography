import torch
import pyro
import pyro.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
import tqdm

from MultitaskVariationalStrategy import MultitaskVariationalStrategy


class MicroGP(pyro.nn.PyroModule):
    def __init__(self,
                 n_latents_env=None,
                 n_variables=None,
                 n_inducing_points_env=None,
                 n_latents_spatial=None,
                 n_inducing_points_spatial=None,
                 unique_coordinates=None,
                 environment=True,
                 spatial=True,
                 traits=True):
        super().__init__()

        self.environment = environment
        self.spatial = spatial
        self.traits = traits

        assert self.environment + self.spatial + self.traits, f"Model cannot run without any components! {self.environment=}, {self.spatial =}, {self.traits=}"
        print(f"Running with components: {self.environment=}, {self.spatial =}, {self.traits=}")

        if self.environment:
            self.n_latents_env = n_latents_env
            self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables, n_inducing_points=n_inducing_points_env)

        if self.spatial:
            self.n_latents_spatial = n_latents_spatial
            self.eta = SpatialGP(n_latents=n_latents_spatial, unique_coordinates=unique_coordinates, n_inducing_points=n_inducing_points_spatial)

    def model(self, batch):
        pyro.module("model", self)

        n_samples = batch.get("n_samples_batch")
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        z = 0

        if self.environment:
            latents_plate = pyro.plate(name="latents_plate", size=self.n_latents_env, dim=-2)

            f_dist = self.f.pyro_model(batch.get("X"), name_prefix="f_GP")

            # Use a plate here to mark conditional independencies
            with pyro.plate(".data_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(dim=0).reshape(n_samples, self.n_latents_env)

            with latents_plate, species_plate:
                w = pyro.sample("w", dist.Normal(loc=torch.zeros(self.n_latents_env, n_species),
                                                 scale=torch.ones(self.n_latents_env, n_species)))

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            z = z + f_samples @ w

        if self.spatial:
            eta_dist = self.eta.pyro_model(batch.get("coords"), name_prefix="eta_GP")

            with pyro.plate(".eta_data_plate", dim=-1):
                # Sample from latent function distribution
                eta_samples = pyro.sample(".eta(coords)", eta_dist)

            eta_samples = eta_samples if eta_samples.shape == torch.Size([batch["n_locs_batch"], self.n_latents_spatial]) else eta_samples.mean(dim=0).reshape(batch["n_locs_batch"], self.n_latents_spatial)
            eta_samples = eta_samples[batch["batch_inverse"]]

            v = pyro.param("v", torch.randn(self.n_latents_spatial, n_species))

            z = z + eta_samples @ v

        if self.traits:
            gamma = pyro.param("gamma", torch.zeros(n_traits))

            bias_loc = batch.get("traits") @ gamma
            bias_scale = torch.ones(n_species)

            with species_plate:
                bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

            # num_particles creates extra samples
            bias = bias if bias.shape == torch.Size([n_species]) else bias.mean(dim=0).squeeze()

            z = z + bias

        with samples_plate, species_plate:
            pyro.sample("y", dist.Bernoulli(logits=z), obs=batch.get("Y") if batch.get("training", True) else None)

    def guide(self, batch):
        n_species = batch.get("n_species")
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)


        if self.environment:
            latents_plate = pyro.plate(name="latents_plate", size=self.n_latents_env, dim=-2)

            w_loc = pyro.param("w_loc", torch.zeros(self.n_latents_env, n_species))
            w_scale = pyro.param("w_scale", torch.ones(self.n_latents_env, n_species), constraint=dist.constraints.positive)

            with latents_plate, species_plate:
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))

            # pyro.module(self.name_prefixes[i], self.gp_models[i])
            f_dist = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate(".data_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

        if self.spatial:
            eta_dist = self.eta.pyro_guide(batch.get("coords"), name_prefix="eta_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate(".eta_data_plate", dim=-1):
                # Sample from latent function distribution
                eta_samples = pyro.sample(".eta(coords)", eta_dist)

        if self.traits:
            bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
            bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)

            with species_plate:
                bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

    def forward(self, x):
        ...


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
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=100),
                batch_shape=torch.Size([n_latents]),
                ard_num_dims=n_variables,
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=25),
            batch_shape=torch.Size([n_latents])
        )

        self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, n_variables)
        self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

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
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            HaversineRBFKernel(  # CustomSpatialKernel(#HaversineRBFKernel(#HaversineRBFKernel(#
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=5),
                batch_shape=torch.Size([n_latents]),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=1),
            batch_shape=torch.Size([n_latents])
        )

        self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, 1) * 5
        self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from development.DataSampler import DataSampler

    from configs.config import config
    from development.misc.save_results import save_results
    from development.misc.calculate_metrics import calculate_metrics

    from sklearn import metrics

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
    traits_path = config["data"]["traits_path"]

    n_latents_env = config["environmental"]["n_latents"]
    n_latents_spatial = config["spatial"]["n_latents"]
    n_iter = config["general"]["n_iter"]
    n_particles = config["general"]["n_particles"]
    device = config["general"]["device"]
    lr = config["general"]["lr"]
    batch_size = config["general"]["batch_size"]
    train_pct = config["general"]["train_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]
    # STOP ARGUMENTS

    dataset = DataSampler(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    n_traits = dataset.n_traits
    unique_coordinates = dataset.unique_coords

    # dataloader = DataLoader(dataset=dataset, batch_size=_batch_size, shuffle=True)
    train_size = int(train_pct * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = MicroGP(
        n_latents_env,
        n_variables,
        n_inducing_points_env,
        n_latents_spatial,
        n_inducing_points_spatial,
        unique_coordinates,
        environment=environment,
        spatial=spatial,
        traits=traits
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    lengthscale_trace = []
    variance_trace = []

    losses = []
    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)
            if i >= 100:
                batch["do_spatial"] = True
            loss += svi.step(batch) / batch.get("Y").nelement()
        losses.append(loss / len(train_dataloader))
        iterator.set_postfix(loss=loss)

        # lengthscale_trace.append(list(model.f.covar_module.base_kernel.lengthscale.squeeze()))
        # variance_trace.append(list(model.f.covar_module.outputscale))

    plt.plot(torch.arange(n_iter), losses)
    plt.show()

    # plt.plot(torch.arange(n_iter), torch.tensor(lengthscale_trace))
    # plt.title('GPyTorch: Lengthscale Trace')
    # plt.show()
    #
    # plt.plot(torch.arange(n_iter), torch.tensor(variance_trace))
    # plt.title('GPyTorch: Variance Trace')
    # plt.show()

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    y_prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        batch = test_dataset.dataset.get_batch_data(idx)
        batch["training"] = False
        batch["do_spatial"] = True

        predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=50)
        y_prob = predictive(batch)["y"].mean(dim=0)
        y_prob_list.append(y_prob)

        y_test_list.append(batch.get("Y"))

    y_prob = torch.concat(y_prob_list)
    test_Y = torch.concat(y_test_list)
    del y_prob_list, y_test_list

    auc_per_species = [
        metrics.roc_auc_score(test_Y[:, i], y_prob[:, i]) if not all(
            test_Y[:, i] == 0) else float("nan") for i in
        range(test_Y.shape[1])
    ]

    auc = torch.tensor(auc_per_species)
    means_tensor = auc[~torch.isnan(auc)]

    if True:  # Histogram
        bin_edges = [round(i * 0.1, 2) for i in range(11)]
        n, bins, patches = plt.hist(means_tensor, bins=bin_edges, color='blue', alpha=0.7, edgecolor='black',
                                    weights=(torch.ones_like(means_tensor) / len(means_tensor) * 100))
        plt.title('Histogram of ROC AUC – GPyTorch S ~ GP')
        plt.xlabel('ROC AUC Bar')
        plt.ylabel('Percentage')
        # Add horizontal grid lines
        for y in range(5, int(max(n)) + 5, 5):
            plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        plt.axvline(x=0.5, color='red', linestyle='-', linewidth=0.8, alpha=0.7)
        plt.show()
        plt.clf()

    metric_results = calculate_metrics(test_Y, y_prob)

    print(metric_results)

    if "cp68wp" in x_path:
        save_results("/Users/cp68wp/Documents/GitHub/Biogeography/results/run_results.xlsx", x_path, y_path, coords_path, traits_path, n_latents_env, n_iter, n_particles, device, lr, batch_size, train_pct, n_inducing_points_env, n_species=dataset.n_species, n_samples=dataset.n_samples, n_env=dataset.n_env, n_traits=None, model_name=model._get_name(), note="Notes", auc=metric_results["AUC"], nll=metric_results["NLL"], mae=metric_results["MAE"], path_figs="", path_model="")
