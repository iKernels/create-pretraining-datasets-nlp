

def read_dataset(dataset):
    r""" Read dataset line-by-line without loading in memory. """
    for line in dataset['train']:
        yield line['text']
