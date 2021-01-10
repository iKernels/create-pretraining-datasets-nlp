MODES = ['one-doc-per-line', 'one-sentence-per-line']


def read_dataset(dataset, mode=MODES[0], limit=None):
    r""" Read dataset line-by-line without loading in memory. """
    if not mode in MODES:
        raise ValueError(
            f"mode {mode} not in allowed {MODES}"
        )

    if mode == MODES[0]:
        for line in dataset['train']:
            yield line['text']

    elif mode == MODES[1]:
        accumulated = 0
        document = ""
        for line in dataset['train']:
            text = line['text']
            if not text.strip() or accumulated >= limit:
                yield document
                accumulated = 0
                document = ""
            else:
                document += text
                accumulated += 1
        if document:
            yield document

    else:
        raise ValueError(
            f"mode {mode} not available"
        )
