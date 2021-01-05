import datasets


def prepare_datasets(parsed_names):
    r"""
    Load the given datasets lazily.
    """
    for name, config in parsed_names:
        yield datasets.load_dataset(name, config)
