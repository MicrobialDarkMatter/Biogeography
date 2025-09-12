import torch


def ndcg_at_k(y_true: torch.Tensor, y_score: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute NDCG@k for predictions.

    Args:
        y_true: relevance labels (batch_size, n_items)
        y_score: predicted scores (batch_size, n_items)
        k: cutoff

    Returns:
        ndcg: tensor of shape (batch_size,)
    """
    # sort by predicted score
    _, indices = torch.topk(y_score, k, dim=1)

    # get true relevance at top-k
    gains = torch.gather(y_true, 1, indices)

    # discounted cumulative gain (DCG)
    discounts = torch.log2(torch.arange(k, device=y_true.device).float() + 2.0)
    dcg = (gains / discounts).sum(dim=1)

    # ideal DCG (IDCG)
    _, ideal_indices = torch.topk(y_true, k, dim=1)
    ideal_gains = torch.gather(y_true, 1, ideal_indices)
    idcg = (ideal_gains / discounts).sum(dim=1)

    # avoid division by zero
    ndcg = dcg / torch.clamp(idcg, min=1e-8)
    return ndcg


def precision_at_k(y_true: torch.Tensor, y_score: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute Precision@k for predictions.

    Args:
        y_true: relevance labels (batch_size, n_items) [0/1 or graded relevance]
        y_score: predicted scores (batch_size, n_items)
        k: cutoff

    Returns:
        precision: tensor of shape (batch_size,)
    """
    # top-k indices by predicted score
    indices_score = torch.topk(y_score, k, dim=1)[1]
    indices_true = torch.topk(y_true, k, dim=1)[1]

    counts = torch.tensor(
        [len(set(a).intersection(set(b))) for a, b in zip(indices_score.tolist(), indices_true.tolist())])
    precision = counts / k
    return precision


def spearman_corr(x, y, dim=1):
    """
    Compute Spearman rank correlation between two tensors along a given dimension.

    Args:
        x, y: Tensors of the same shape
        dim: Dimension along which to compute correlation

    Returns:
        Spearman correlation tensor
    """
    # Rank along the given dimension
    x_rank = torch.argsort(torch.argsort(x, dim=dim), dim=dim).float()
    y_rank = torch.argsort(torch.argsort(y, dim=dim), dim=dim).float()

    # Mean centering
    x_rank = x_rank - x_rank.mean(dim=dim, keepdim=True)
    y_rank = y_rank - y_rank.mean(dim=dim, keepdim=True)

    # Compute Pearson correlation on ranks
    cov = (x_rank * y_rank).sum(dim=dim)
    x_std = torch.sqrt((x_rank ** 2).sum(dim=dim))
    y_std = torch.sqrt((y_rank ** 2).sum(dim=dim))

    return cov / (x_std * y_std)


def rmse(pred, target, dim=1):
    """
    Compute Root Mean Squared Error (RMSE) between pred and target.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimension(s) to reduce over, if None reduces all elements

    Returns:
        RMSE value or tensor along specified dimension
    """
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=dim))
