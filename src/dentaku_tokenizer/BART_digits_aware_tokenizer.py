import copy
import dataclasses
import logging
from typing import Callable, List, Optional, Union

from .dataset_utils import convert_word_to_number
from .util import split_tokens_by_hyphen


from allennlp.data.tokenizers import (PretrainedTransformerTokenizer,
                                      SpacyTokenizer, Token, Tokenizer)

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def split_digits_into_chars(
    tokens: List[Token],
    convert_token_to_ids: Optional[Callable[[str], int]] = None,
    unk_id: Optional[int] = None,
) -> List[Token]:
    """
    conversion is done as follows depending on whether the tokenizer is BART:
    BART: "Ġ100" --> ["Ġ1", "0", "0"]
    """

    digits = set("Ġ0123456789")
    new_tokens = []
    for token in tokens:
        chars = set(token.text)
        if chars.issubset(digits) and chars != {"Ġ"}:
            for index, digit in enumerate(token.text.replace("Ġ", "")):
                if token.text.startswith("Ġ") and index == 0:
                    digit = "Ġ" + digit
                token = Token(digit)
                if convert_token_to_ids is not None:
                    token.text_id = convert_token_to_ids(digit)
                    assert unk_id is None or token.text_id != unk_id
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return new_tokens


class DigitsAwareTransformerTokenizer(object):
    """
    >>> In [1]: tokenizer = DigitsAwareTransformerTokenizer('bert-base-uncased')
    >>> In [2]: tokenizer.tokenize("Do you have 100 dollars?")
    >>>  [Token(text='I', text_id=100, type_id=0, idx=0, idx_end=1),
    >>>  Token(text='Ġhave', text_id=33, type_id=0, idx=1, idx_end=5),
    >>>  Token(text='Ġ1', text_id=112, type_id=None, idx=None, idx_end=None),
    >>>  Token(text='2', text_id=176, type_id=None, idx=None, idx_end=None),
    >>>  Token(text='3', text_id=246, type_id=None, idx=None, idx_end=None),
    >>>  Token(text='4', text_id=306, type_id=None, idx=None, idx_end=None),
    >>>  Token(text='5', text_id=245, type_id=None, idx=None, idx_end=None),
    >>>  Token(text='Ġdollars', text_id=1932, type_id=0, idx=1, idx_end=8),
    >>>  Token(text='.', text_id=4, type_id=0, idx=0, idx_end=1)]
    """

    def __init__(
        self, transformer_model_name='facebook/bart-base', include_more_numbers: bool = False
    ) -> None:

        self.transformer_model_name = transformer_model_name
        self._word_tokenizer = SpacyTokenizer()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_model_name,
            add_special_tokens=False,
            do_lower_case=True,
            is_fast=False,
        )

        self.include_more_numbers = include_more_numbers
        self._unk_id = self.convert_tokens_to_ids(self.tokenizer.unk_token)

        try:
            self._reverse_engineer_special_tokens("a", "b")
        except AssertionError:
            # For most transformer models, "a" and "b" work just fine as dummy tokens.  For a few,
            # they don't, and so we use "1" and "2" instead.
            self._reverse_engineer_special_tokens("1", "2")

    def _reverse_engineer_special_tokens(
        self,
        token_a: str,
        token_b: str,
    ):
        # storing the special tokens
        self.sequence_pair_start_tokens = []
        self.sequence_pair_mid_tokens = []
        self.sequence_pair_end_tokens = []
        # storing token type ids for the sequences
        self.sequence_pair_first_token_type_id = None
        self.sequence_pair_second_token_type_id = None

        # storing the special tokens
        self.single_sequence_start_tokens = []
        self.single_sequence_end_tokens = []
        # storing token type id for the sequence
        self.single_sequence_token_type_id = None

        # Reverse-engineer the tokenizer for two sequences
        tokenizer_with_special_tokens = AutoTokenizer.from_pretrained(
            self.transformer_model_name,
            add_special_tokens=True,
            do_lower_case=True,
        )
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            token_b,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [
                0] * len(dummy_output["input_ids"])

        dummy_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        assert dummy_a in dummy_output["input_ids"]
        dummy_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]
        assert dummy_b in dummy_output["input_ids"]
        assert dummy_a != dummy_b

        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_a = True
                assert (
                    self.sequence_pair_first_token_type_id is None
                    or self.sequence_pair_first_token_type_id == token_type_id
                ), "multiple different token type ids found for the first sequence"
                self.sequence_pair_first_token_type_id = token_type_id
                continue

            if token_id == dummy_b:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_b = True
                assert (
                    self.sequence_pair_second_token_type_id is None
                    or self.sequence_pair_second_token_type_id == token_type_id
                ), "multiple different token type ids found for the second sequence"
                self.sequence_pair_second_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)

        assert (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=True)

        # Reverse-engineer the tokenizer for one sequence
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )

        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [
                0] * len(dummy_output["input_ids"])

        seen_dummy_a = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_a = True
                assert (
                    self.single_sequence_token_type_id is None
                    or self.single_sequence_token_type_id == token_type_id
                ), "multiple different token type ids found for the sequence"
                self.single_sequence_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)

        assert (
            len(self.single_sequence_start_tokens)
            + len(self.single_sequence_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=False)

    def _tokenize(self, text: str, **kwargs) -> List[Token]:

        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=None,
            truncation=False,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
            **kwargs
        )

        token_ids, token_type_ids, special_tokens_mask, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens["special_tokens_mask"],
            encoded_tokens["offset_mapping"],
        )

        tokens = []
        for token_id, token_type_id, special_token_mask, (start, end) in zip(
            token_ids, token_type_ids, special_tokens_mask, token_offsets
        ):
            if special_token_mask == 1:
                continue

            tokens.append(
                Token(
                    text=self.tokenizer.convert_ids_to_tokens(
                        token_id, skip_special_tokens=False
                    ),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )

        return tokens

    # 多分strの時にしか使えない
    """
    def convert_tokens_to_ids(
        self, tokens: Union[Token, List[Token]]
    ) -> Union[Token, List[Token]]:
        result = self.tokenizer.convert_tokens_to_ids(tokens)
        return result
    """

    # 型アノテーションを変更
    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        result = self.tokenizer.convert_tokens_to_ids(tokens)
        return result

    def normalize_digits(
        self,
        tokens: List[Token],
    ) -> List[Token]:
        new_tokens = []
        for token in tokens:
            number = convert_word_to_number(
                token.text, self.include_more_numbers)
            if number is not None:
                new_tokens.append(dataclasses.replace(token, text=str(number)))
            else:
                new_tokens.append(token)
        return new_tokens

    def str_to_tokens(self, text) -> List[Token]:
        tokens = self._word_tokenizer.tokenize(text)
        tokens = split_tokens_by_hyphen(tokens)
        tokens = self.normalize_digits(tokens)
        return tokens

    def tokens_to_wordpieces(
        self,
        tokens: List[Token],
        skip_tokens: List[Token] = None,
    ) -> List[Token]:
        def split_token_to_wordpieces(text) -> List[Token]:
            wordpieces = self._tokenize(text)
            wordpieces = split_digits_into_chars(
                wordpieces, self.convert_tokens_to_ids, self._unk_id
            )
            return wordpieces

        new_tokens = []
        for token in tokens:
            if skip_tokens is not None and token in skip_tokens:
                new_tokens.append(token)
            elif len(new_tokens) > 0 and new_tokens[-1].idx_end != token.idx:
                # a bit of hack so Ġ is prepended to the first wordpiece
                # cf. https://huggingface.co/transformers/_modules/transformers/models/roberta/tokenization_roberta.html#RobertaTokenizer
                new_tokens.extend(split_token_to_wordpieces(" " + token.text))
            else:
                new_tokens.extend(split_token_to_wordpieces(token.text))
        return new_tokens

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.str_to_tokens(text)
        wordpieces = self.tokens_to_wordpieces(tokens)
        return wordpieces

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]

        # Make sure we don't change the input parameters
        tokens2 = copy.deepcopy(tokens2)

        if tokens2 is None:
            return (
                self.single_sequence_start_tokens
                + with_new_type_id(tokens1, self.single_sequence_token_type_id)
                + self.single_sequence_end_tokens
            )
        else:
            return (
                self.sequence_pair_start_tokens
                + with_new_type_id(tokens1,
                                   self.sequence_pair_first_token_type_id)
                + self.sequence_pair_mid_tokens
                + with_new_type_id(tokens2,
                                   self.sequence_pair_second_token_type_id)
                + self.sequence_pair_end_tokens
            )

    def convert_ids_to_tokens(
            self, indexes: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        return self.tokenizer.convert_ids_to_tokens(indexes)


if __name__ == "__main__":
    tokenizer = DigitsAwareTransformerTokenizer(
        "facebook/bart-base", include_more_numbers=True
    )
    print("unk_id:", tokenizer._unk_id)
    print("unk_token:", tokenizer.tokenizer.unk_token)
    result = tokenizer.tokenize("I have 12345 dollars.")
    for token in result:
        print(token)
