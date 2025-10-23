import pyro

from ExploreSpatial.GPyTorch_GP import MultitaskGPModel


class SpatialModel(pyro.nn.PyroModule):
    def __init__(self, n_latents, n_species, inducing_points):
        self.n_latents = n_latents
        super().__init__()

        self.eta = MultitaskGPModel(num_latents=self.n_latents, num_tasks=n_species, inducing_points=inducing_points)

    # @pyro.poutine.scale(scale=1.0 / N_data)
    def model(self, feature, batch_inverse, Y, training):
        pyro.module("model", self)

        eta_dist = self.eta.pyro_model(feature, name_prefix="eta_GP")

        with pyro.plate("eta_data_plate", size=feature.shape[0]):  # , dim=-1):
            # Sample from latent function distribution
            eta_samples = pyro.sample("eta(feature)", eta_dist)
        # eta_samples = pyro.sample("eta(feature)", eta_dist.to_event(1))

        samples_plate = pyro.plate(name="samples_plate", size=Y.shape[0], dim=-2)
        species_plate = pyro.plate(name="species_plate", size=Y.shape[1], dim=-1)

        scales = torch.ones_like(eta_samples) * 0.1

        with samples_plate, species_plate:
            # pyro.sample("y", dist.Bernoulli(logits=eta_samples), obs=Y if training else None)
            pyro.sample("y", dist.Normal(loc=eta_samples, scale=scales), obs=Y if training else None)

    # @pyro.poutine.scale(scale=1.0 / N_data)
    def guide(self, feature, batch_inverse, Y, training):
        pyro.module("guide", self)
        eta_dist = self.eta.pyro_guide(feature, name_prefix="eta_GP")
        # Use a plate here to mark conditional independencies
        with pyro.plate("eta_data_plate", size=feature.shape[0]):  # , dim=-1):
            # Sample from latent function distribution
            eta_samples = pyro.sample("eta(feature)", eta_dist)
        # eta_samples = pyro.sample("eta(feature)", eta_dist.to_event(1))

    def forward(self, feature, batch_inverse, Y, training):
        ...
        # self.model(feature, batch_inverse, Y, training)


if __name__ == "__main__":
    from ExploreSpatial.load_data import *

    #########
    # Model #
    #########
    model = SpatialModel(n_latents=num_latents, n_species=J, inducing_points=inducing_points)

    optimizer = pyro.optim.Adam({"lr": 0.01, "betas": (0.99, 0.999)})  # TODO: Learning Rate
    elbo = pyro.infer.TraceMeanField_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    ############
    # Training #
    ############
    points = [model.eta.variational_strategy.base_variational_strategy.inducing_points.clone().detach()]

    losses = []

    iterator = tqdm.tqdm(range(200))  # TODO: Number of interation
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)

            feature = batch.get(feature_input)
            batch_inverse = batch.get("batch_inverse")
            Y = batch.get("Y")
            training = batch.get("training", True)

            loss += svi.step(feature, batch_inverse, Y, training)  # / len(feature)

        iterator.set_postfix(loss=loss)

        points.append(model.eta.variational_strategy.base_variational_strategy.inducing_points.clone().detach())

        losses.append(loss)

    points = torch.stack(points)

    plt.plot(losses)
    plt.show()

    ###########
    # Testing #
    ###########
    y_prob_list = []
    y_test_list = []

    for idx in test_dataloader:
        batch = dataset.get_batch_data(idx)
        batch["training"] = False
        batch["do_spatial"] = True

        feature = batch.get(feature_input)
        batch_inverse = batch.get("batch_inverse")
        Y = batch.get("Y")
        training = batch.get("training", True)

        predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=50)
        y_prob = predictive(feature, batch_inverse, Y, training)["y"].mean(dim=0)
        y_prob_list.append(y_prob)

        y_test_list.append(batch.get("Y"))

    y_prob = torch.concat(y_prob_list)
    test_Y = torch.concat(y_test_list)
    del y_prob_list, y_test_list

    metric_results = calculate_metrics(test_Y, y_prob)

    print("ROC AUC: ", metric_results)

    ##################################
    # Plotting Inducing Points Trace #
    ##################################
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor='white')
    axes = axes.flatten()

    steps, _, n_params, dim = points.shape

    for j in range(num_latents):
        params = points[:, j, :, :].detach()
        ax = axes[j]

        for i in range(n_params):
            # Build line segments for each trajectory
            x = params[:, i, 0]
            y = params[:, i, 1]
            points_ = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points_[:-1], points_[1:]], axis=1)

            # Define color gradient along steps
            colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))

            # Create the colored line
            lc = LineCollection(segments, colors=colors, linewidth=1.5, alpha=0.8)
            ax.add_collection(lc)

            # Mark start and end points
            ax.scatter(x[0], y[0], color='lightgreen', s=30, zorder=3)
            ax.scatter(x[-1], y[-1], color='red', s=30, zorder=3)

        ax.set_title(f"Latent {j + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.autoscale()  # adjust view to data

    # Hide unused subplots
    for j in range(num_latents, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
