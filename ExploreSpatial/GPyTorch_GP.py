import gpytorch
import torch


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, inducing_points):
        """

        :param num_latents:
        :param num_tasks:
        :param inducing_points: Shape = [num_latents, num_points, num_features]
        """
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.MaternKernel(
            lengthscale_prior=gpytorch.priors.NormalPrior(5, 5),
            batch_shape=torch.Size([num_latents]),
            ard_num_dims=inducing_points.shape[-1]
        )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(
        #         lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=25),
        #         batch_shape=torch.Size([num_latents]),
        #     ),
        #     outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=1),
        #     batch_shape=torch.Size([num_latents])
        # )

        # self.covar_module.lengthscale = torch.rand(num_latents, 1, 1, dtype=torch.float32) * 0.1

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    from ExploreSpatial.load_data import *

    pyro.clear_param_store()

    model = MultitaskGPModel(num_latents, num_tasks, inducing_points)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    # likelihood = gpytorch.likelihoods.BernoulliLikelihood()  # Does not work!

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=I)

    ############
    # TRAINING #
    ############
    points = [model.variational_strategy.base_variational_strategy.inducing_points.clone().detach()]

    losses = []
    # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
    # effective for VI.
    epochs_iter = tqdm.tqdm(range(n_iter), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        optimizer.zero_grad()
        output = model(features)
        loss = -mll(output, Y)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        points.append(model.variational_strategy.base_variational_strategy.inducing_points.clone().detach())

    points = torch.stack(points)

    plt.plot(losses)
    plt.show()

    ###########
    # Testing #
    ###########

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        # lower, upper = predictions.confidence_region()

    print("ROC AUC: ", calculate_metrics(test_y, mean.clamp(min=0, max=1)))

    ##################################
    # Plotting Inducing Points Trace #
    ##################################
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), facecolor='white')

    axes = axes.flatten()  # make indexing easy

    steps, _, n_params, dim = points.shape

    for j in range(num_latents):
        params = points[:, j, :, :].detach()
        ax = axes[j]

        for i in range(n_params):
            ax.scatter(params[-1, i, 0], params[-1, i, 1], color='red', s=30)  # end point
            ax.plot(params[:, i, 0], params[:, i, 1], alpha=0.7, lw=1.5)
            ax.scatter(params[0, i, 0], params[0, i, 1], color='lightgreen', s=30)  # start point

        ax.set_title(f"Latent {j + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Hide unused subplots if num_latents < rows*cols
    for j in range(num_latents, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
