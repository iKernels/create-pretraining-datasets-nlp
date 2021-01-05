from blingfire import text_to_sentences
from tqdm import tqdm


def documents_to_sentences(documents, limit=None):
    r""" Split each document in a sentence using the blingfire tokenizer. """

    for i, document in tqdm(enumerate(documents), desc="Splitting docs in sentences"):

        if limit is not None and i >= limit:
            break

        sentences = text_to_sentences(document)
        sentences = sentences.split("\n")

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                yield sentence
        yield ""
