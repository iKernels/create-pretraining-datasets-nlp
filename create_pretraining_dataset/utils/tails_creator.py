from transformers import (
    AutoTokenizer,
    BertTokenizer, BertTokenizerFast,
    RobertaTokenizer, RobertaTokenizerFast,
    XLNetTokenizer, XLNetTokenizerFast,
    GPT2Tokenizer, GPT2TokenizerFast,
    ElectraTokenizer, ElectraTokenizerFast
)


class TailsCreator:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

        if isinstance(
            tokenizer, (BertTokenizer, BertTokenizerFast, ElectraTokenizer, ElectraTokenizerFast)
        ):
            self.get_words_tails = self.bert_word_tails

        elif isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.get_words_tails = self.roberta_words_tails

        elif isinstance(tokenizer, (XLNetTokenizer, XLNetTokenizerFast)):
            self.get_words_tails = self.xlnet_words_tails

        elif isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            self.get_words_tails = self.gpt_words_tails
  
    def bert_word_tails(self, ids):
        return [
            token.startswith('##') for token in self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        ]

    def roberta_words_tails(self, ids):
        return [
            (not token.startswith('Ġ')) and (token not in self.tokenizer.all_special_tokens) and (i != 1)
            for i, token in enumerate(self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False))
        ]

    def gpt_words_tails(self, ids):
        raise NotImplementedError("This tokenizer is not supported yet")

    def xlnet_words_tails(self, ids):
        raise NotImplementedError("This tokenizer is not supported yet")
