def flatten_dict(d, parent_key="", sep="_"):
    """
    Recursively flattens a nested dictionary.

    Example:
    {"data": {"A": 1, "B": 2}} → {"data_A": 1, "data_B": 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
