from blingfire import text_to_sentences
from tqdm import tqdm

MODES = ['one-doc-per-line', 'one-sentence-per-line']


def dataset_to_sentences(documents, mode=MODES[0], limit=None, limit_sentences_per_doc=None):
    r"""
    This function taken in input a dataset instance and returns a sequence of sentences with some empty sentences
    separating different documents.
    
    Return:
        Generator creating a sequence like: "sentence_1_doc_1", "sentence_2_doc_1", "sentence_3_doc_1", "", "sentence_1_doc_2", ...
    """

    if not mode in MODES:
        raise ValueError(
            f"mode {mode} not in allowed {MODES}"
        )

    # common pbar arguments
    kwargs = {'total': None} 
    if limit is not None:
        kwargs['total'] = limit
    
    if mode == MODES[0]:
        r""" Dataset as sequence of documents. """

        if limit is None:
            try:
                kwargs['total'] = documents.num_rows
            except:
                kwargs['total'] = len(documents)

        for i, document in tqdm(enumerate(documents), desc="(Splitting) Splitting documents in sentences", position=0, **kwargs):

            if limit is not None and i >= limit:
                break

            sentences = text_to_sentences(document['text'])
            sentences = sentences.split("\n")

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    yield sentence
            yield ""

    elif mode == MODES[1]:
        r"""
        Read sentences and insert documents separators if needed
        after having seen `limit_sentences_per_doc` sentences, return a document separator ("")
        """

        accumulated = 0
        documents_returned = 0

        pbar = tqdm(desc="(Splitting) Creating documents from sentences", position=0, **kwargs)

        for sentence in documents:
            
            if limit is not None and documents_returned >= limit:
                break

            if not sentence['text'].strip() or (limit_sentences_per_doc and accumulated >= limit_sentences_per_doc):
                accumulated = 0
                documents_returned += 1
                pbar.update(1)
                yield ""

            else:
                accumulated += 1
                yield sentence['text']

    else:
        raise ValueError(
            f"mode {mode} not available"
        )
