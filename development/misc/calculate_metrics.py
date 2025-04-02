from sklearn import metrics
import torch


def calculate_metrics(y_true, y_pred):
    res = {}

    # AUC
    auc_per_species = [
        metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    auc = torch.tensor(auc_per_species)
    auc = (auc[~torch.isnan(auc)]).mean().item()
    res["AUC"] = auc

    # NLL
    nll_per_species = [
        metrics.log_loss(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    nll = torch.tensor(nll_per_species)
    nll = (nll[~torch.isnan(nll)]).mean().item()
    res["NLL"] = nll

    # MAE
    mae_per_species = [
        metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    mae = torch.tensor(mae_per_species)
    mae = (mae[~torch.isnan(mae)]).mean().item()
    res["MAE"] = mae

    # PR AUC
    pr_auc_per_species = [
        metrics.average_precision_score(y_true[:, i], y_pred[:, i]) if not all(
            y_true[:, i] == 0) else float("nan") for i in
        range(y_true.shape[1])
    ]
    pr_auc = torch.tensor(pr_auc_per_species)
    pr_auc = (pr_auc[~torch.isnan(pr_auc)]).mean().item()
    res["PR_AUC"] = pr_auc

    return res
