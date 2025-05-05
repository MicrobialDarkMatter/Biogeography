import pyro

from hmsc import *

from configs.config import config

from development.misc.calculate_metrics import calculate_metrics

res = {
    "ROC AUC": [],
    "NLL": [],
    "MAE": [],
    "PR AUC": [],

}

for seed in range(40, 45):
    pyro.clear_param_store()

    dataset = CustomDataSubsampling(
        Y_path=config["data"]["Y_path"],
        X_path=config["data"]["X_path"],
        coords_path=config["data"]["coords_path"],
        traits_path=config["data"]["traits_path"],
        device=config["general"]["device"],
        normalize_X=config["data"]["normalize_X"],
        prevalence_threshold=config["data"]["prevalence_threshold"]
    )

    unique_coordinates = dataset.unique_coords if dataset.using_coordinates else torch.rand(2, 2)

    model = HMSC_GP(unique_coordinates=unique_coordinates)

    train_size = int(config["general"]["train_pct"] * len(dataset))
    test_size = len(dataset) - train_size
    train_size = train_size - test_size
    val_size = test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size],
                                                            generator=torch.Generator().manual_seed(seed))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["general"]["batch_size"], shuffle=True)

    # Make sure at least 10 species obserservations are present in each subset of the data
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= 10) & (
            dataset.Y[test_dataset.indices].sum(dim=0) >= 10)
    dataset.Y = dataset.Y[:, keep_y]
    if config["data"]["traits_path"]:
        dataset.traits = dataset.traits[keep_y, :]
    dataset.n_species = dataset.Y.shape[1]

    # # Set up the optimizer
    optimizer = Adam({"lr": config["general"]["lr"]})

    # Training
    if config["hmsc"]["cross_validation"]:
        train_svi_cv(
            k_fold=config["hmsc"]["k_fold"],
            train_dataset=train_dataset,
            batch_size=config["general"]["batch_size"],
            epoch=config["general"]["n_iter"],
            model=model.model,
            guide=model.guide,
            likelihood=config["hmsc"]["likelihood"],
            optimizer=optimizer,
            verbose=config["general"]["verbose"]
        )
    else:
        train_svi(
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            epoch=config["general"]["n_iter"],
            model=model.model,
            guide=model.guide,
            likelihood=config["hmsc"]["likelihood"],
            optimizer=optimizer,
            verbose=config["general"]["verbose"]
        )

    # Testing
    test_idx = test_dataset.indices
    # test_data = dataset.get_batch_data(test_idx)
    test_data = test_dataset.dataset.get_batch_data(test_idx)
    test_data["training"] = False

    predictive = Predictive(model.model, guide=model.guide, num_samples=100)

    predict = predictive(test_data, config["hmsc"]["likelihood"])["y"].mean(dim=0)

    metrics = calculate_metrics(test_data.get("Y"), predict)

    res["ROC AUC"].append(metrics["AUC"])
    res["NLL"].append(metrics["NLL"])
    res["MAE"].append(metrics["MAE"])
    res["PR AUC"].append(metrics["PR_AUC"])

for key, value in res.items():
    print(key, torch.tensor(value).mean())