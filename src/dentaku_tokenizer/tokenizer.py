import argparse
from itertools import chain
from pydoc import splitdoc
from typing import Dict, List, Optional, Tuple, Union
import re
from logzero import logger
from transformers import PreTrainedTokenizer, BartTokenizer, PreTrainedTokenizerFast, BartTokenizerFast, T5Tokenizer


from .BART_digits_aware_tokenizer import DigitsAwareTransformerTokenizer


class T5DentakuTokenizer(T5Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_int(self, str):
        return True if re.fullmatch(r'▁{0,1}[0-9]+', str) else False

    def _tokenize(self, text, **kwargs) -> List[str]:
        tokenized_text = []
        encoded_text = self.sp_model.encode(text, out_type=str)

        for word in encoded_text:
            if self._is_int(word):
                splited_word = list(word)
                tokenized_text = tokenized_text + splited_word
            else:
                tokenized_text.append(word)
        return tokenized_text


class BartDentakuTokenizer(BartTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.name_or_path == 'facebook/bart-base':
            logger.warning(
                "This tokenizer is only supported as \"facebook/bart-base\"!")

        self.digit_tokenizer = DigitsAwareTransformerTokenizer(
            self.name_or_path)

    def _tokenize(self, text, **kwargs) -> List[str]:
        tokenized_text = [
            token.text for token in self.digit_tokenizer.tokenize(text)]
        tokenized_text[0] = "Ġ" + tokenized_text[0]
        return tokenized_text
