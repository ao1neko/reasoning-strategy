from argparse import ArgumentParser
import itertools
from contextlib import contextmanager
from functools import reduce
from operator import mul
from typing import Iterator, List, Optional, Tuple, Union

import math
import numpy
import torch
#from sparsemax import Sparsemax
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.util import (get_text_field_mask,
                              get_token_ids_from_text_field_tensors,
                              min_value_of_dtype)

from .dataset_utils import convert_word_to_number
from .bio import BTAG, ITAG

# I follow the original implementation of using these symbols
# representing START/END_SYMBOL, since bert's vocab doesn't contain `@start@`
START_SYMBOL, END_SYMBOL, SEP, DUMMY = "@", "\\", ";", "!"

NONE, QUESTION, PASSAGE, PADDING = range(4)

ANSWER_GENERATION, ANSWER_SPAN = range(2)


def detokenize(tokens: List[str]) -> List[str]:
    text = " ".join(tokens)

    # De-tokenize WordPieces that have been split off.
    text = text.replace(" ##", "").replace("##", "")

    # Clean whitespace
    return " ".join(text.strip().split())


def post_process_decoded_output(text: str) -> str:
    # remove space around decimal point
    processed = ".".join(x.strip() for x in text.split("."))
    try:
        # '.' is a decimal only if final str is a number
        float(processed)
    except ValueError:
        processed = text

    # remove space around "-"
    result = "-".join(x.strip() for x in processed.split("-"))
    return result


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    """
    This is modified by the original AllenNLP implementation so this
    explicitly specifies `idx_end` field in Token class.
    """
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(
                Token(text=sub_str, idx=char_offset,
                      idx_end=char_offset + len(sub_str))
            )
            char_offset += len(sub_str)
        split_tokens.append(
            Token(text=delimiter, idx=char_offset,
                  idx_end=char_offset + len(delimiter))
        )
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    return [token]


def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    hyphens = ["-", "–", "~"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(
                            unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens


def gather_row(x, indices):
    """Kind of a broadcast version of `torch.gather` function.
    Currently this support for inputs `x` with 3 dimensions and
    `indices` with 2 dimensions.

    Example:
    >>> x = torch.tensor([
    ...     [[1, 2],
    ...      [3, 4]],
    ...     [[5, 6]
    ...      [7, 8]]
    ... ])
    >>> indices = torch.tensor(
    ...     [[0, 0],
    ...      [1, 0]]
    ... )
    >>> gather_row(x, indices)
    torch.tensor([
        [[1, 2],
         [1, 2]]
        [[7, 8],
         [5, 6]]
    ])
    """
    assert (
        len(x.size()) == 3 and len(indices.size()) == 2
    ), "not supported input tensor shape"
    batch_size, sequence_size, hidden_size = x.size()
    indices = indices + torch.arange(
        0, batch_size * sequence_size, sequence_size)\
        .to(x.device)[:, None]

    out = x.view((batch_size * sequence_size, hidden_size))
    out = out.index_select(0, indices.flatten())
    out = out.reshape(indices.size() + (hidden_size,))
    return out


def gumbel_noise(
    shape: torch.Size,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> torch.FloatTensor:
    uniform = torch.rand(shape).to(device)

    return -torch.log(-torch.log(uniform + eps) + eps)


def gumbel_noise_like(tensor: torch.FloatTensor) -> torch.FloatTensor:
    return gumbel_noise(shape=tensor.size(), device=tensor.device)


def onehot(x: torch.Tensor, n: int) -> torch.Tensor:
    x0 = x.view(-1, 1)
    x1 = x.new_zeros(len(x0), n, dtype=x.dtype)
    x1.scatter_(1, x0, 1)
    return x1.view(x.size() + (n,))


def masked_binary_cross_entropy_with_logits(
    inputs: torch.FloatTensor,
    target: torch.FloatTensor,
    mask: torch.BoolTensor,
    only_positive_instances: bool = False,
) -> torch.FloatTensor:

    if not only_positive_instances:
        output = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, target, reduction="none"
        )
    else:
        output = - target * torch.nn.functional.logsigmoid(inputs)

    return (output * mask.float()).sum(dim=-1).mean()


def binary_entropy(
    inputs: torch.FloatTensor,
    mask: torch.BoolTensor,
    eps: float = 1e-12,
) -> torch.FloatTensor:
    neg = 1 - inputs
    results = (
        - inputs * torch.log(inputs + eps)
        - neg * torch.log(neg + eps)
    )
    return (results * mask.float()).sum(dim=-1).mean()


"""
def masked_sparsemax(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int = -1,
) -> torch.Tensor:
    if mask is None:
        result = Sparsemax(dim=dim)(vector)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill(
            ~mask, min_value_of_dtype(vector.dtype))
        result = Sparsemax(dim=dim)(masked_vector)
    return result
"""


def masked_argmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(
        ~mask, min_value_of_dtype(vector.dtype))
    _, max_indices = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_indices


def left_align_mask(mask: torch.Tensor, reduce_size: bool = False) -> torch.Tensor:
    """
    >>> mask
    ... tensor([[1, 1, 0, 0, 0],
    ...         [0, 1, 1, 1, 0],
    ...         [0, 0, 0, 0, 1],
    ...         [0, 0, 0, 1, 0]])

    >>> left_align_mask(mask)
    ... tensor([[1, 1, 0, 0, 0],
    ...         [1, 1, 1, 0, 0],
    ...         [1, 0, 0, 0, 0],
    ...         [1, 0, 0, 0, 0]])
    """

    assert len(mask.size()) == 2

    _, seq_len = mask.size()
    lengths = mask.long().sum(dim=1)

    if reduce_size:
        seq_len = lengths.max().item()

    new_mask = torch.tril(
        mask.new_ones(seq_len + 1, seq_len, dtype=mask.dtype), diagonal=-1
    )[lengths]

    return new_mask


def make_mask_of_lengths(
    n: Union[torch.LongTensor, List[int]], device: Optional[torch.device] = None
) -> torch.BoolTensor:

    if isinstance(n, list):
        n = torch.tensor(n, device=device, dtype=torch.long)

    assert len(n.size()) == 1

    seq_len = n.max().item()
    shape = (seq_len + 1, seq_len)
    mask = torch.tril(n.new_ones(shape, dtype=torch.bool), diagonal=-1)[n]
    return mask


def masked_embedding_lookup_with_padding(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding: Union[int, float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    >>> embeddings # (4, 5, 8)
    ... tensor([[[  0,   1,   2,   3,   4,   5,   6,   7],
    ...          [  8,   9,  10,  11,  12,  13,  14,  15],
    ...          [ 16,  17,  18,  19,  20,  21,  22,  23],
    ...          [ 24,  25,  26,  27,  28,  29,  30,  31],
    ...          [ 32,  33,  34,  35,  36,  37,  38,  39]],
    ...
    ...         [[ 40,  41,  42,  43,  44,  45,  46,  47],
    ...          [ 48,  49,  50,  51,  52,  53,  54,  55],
    ...          [ 56,  57,  58,  59,  60,  61,  62,  63],
    ...          [ 64,  65,  66,  67,  68,  69,  70,  71],
    ...          [ 72,  73,  74,  75,  76,  77,  78,  79]],
    ...
    ...         [[ 80,  81,  82,  83,  84,  85,  86,  87],
    ...          [ 88,  89,  90,  91,  92,  93,  94,  95],
    ...          [ 96,  97,  98,  99, 100, 101, 102, 103],
    ...          [104, 105, 106, 107, 108, 109, 110, 111],
    ...          [112, 113, 114, 115, 116, 117, 118, 119]],
    ...
    ...         [[120, 121, 122, 123, 124, 125, 126, 127],
    ...          [128, 129, 130, 131, 132, 133, 134, 135],
    ...          [136, 137, 138, 139, 140, 141, 142, 143],
    ...          [144, 145, 146, 147, 148, 149, 150, 151],
    ...          [152, 153, 154, 155, 156, 157, 158, 159]]])

    >>> mask
    ... tensor([[1, 1, 0, 0, 0],
    ...         [0, 1, 1, 1, 0],
    ...         [0, 0, 0, 0, 1],
    ...         [0, 0, 0, 1, 0]])

    >>> masked_embedding_lookup_with_padding(embeddings, mask)
    ... tensor([[[  0,   1,   2,   3,   4,   5,   6,   7],
    ...          [  8,   9,  10,  11,  12,  13,  14,  15],
    ...          [  0,   0,   0,   0,   0,   0,   0,   0]],
    ...
    ...         [[ 48,  49,  50,  51,  52,  53,  54,  55],
    ...          [ 56,  57,  58,  59,  60,  61,  62,  63],
    ...          [ 64,  65,  66,  67,  68,  69,  70,  71]],
    ...
    ...         [[112, 113, 114, 115, 116, 117, 118, 119],
    ...          [  0,   0,   0,   0,   0,   0,   0,   0],
    ...          [  0,   0,   0,   0,   0,   0,   0,   0]],
    ...
    ...         [[144, 145, 146, 147, 148, 149, 150, 151],
    ...          [  0,   0,   0,   0,   0,   0,   0,   0],
    ...          [  0,   0,   0,   0,   0,   0,   0,   0]]])
    """
    assert embeddings.size()[:-1] == mask.size()

    assert isinstance(padding, (int, float, torch.Tensor))

    assert (
        not isinstance(padding, torch.Tensor)
        or len(padding.size()) == 1
        and embeddings.size(-1) == padding.size(0)
    )

    assert torch.max(mask) <= 1

    batch_size, _, embedding_size = embeddings.size()
    device = embeddings.device

    to_mask = left_align_mask(mask.bool(), reduce_size=True)

    seq_len = to_mask.long().sum(dim=1).max().item()

    if isinstance(padding, torch.Tensor):
        paddings = padding.unsqueeze(0).unsqueeze(
            1).repeat(batch_size, seq_len, 1)
    else:
        paddings = torch.full(
            (batch_size, seq_len, embedding_size),
            fill_value=padding,
            dtype=torch.float,
            device=device,
        )

    paddings.masked_scatter_(to_mask.unsqueeze(-1), embeddings[mask.bool()])

    return paddings, to_mask


def get_one_embedding_vector_of(
    embedder_or_embedding: Union[TextFieldEmbedder, torch.nn.Embedding],
    index: int = 0,
    namespace: str = "tokens",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    assert isinstance(embedder_or_embedding,
                      (TextFieldEmbedder, torch.nn.Embedding))

    onehot_tensor = torch.tensor([index], dtype=torch.long, device=device)

    if isinstance(embedder_or_embedding, TextFieldEmbedder):
        text_field_tensors = {
            namespace: {"tokens": onehot_tensor}
        }  # this is too simple
        return embedder_or_embedding(text_field_tensors).squeeze(0)
    return embedder_or_embedding(onehot_tensor).squeeze(0)


def mask_mask_copy(
    values: torch.FloatTensor,
    from_mask: torch.BoolTensor,
    to_mask: torch.BoolTensor,
    default_value: float = 0.0,
) -> torch.FloatTensor:

    assert from_mask.long().sum() == to_mask.long().sum()

    result = torch.full_like(
        to_mask, fill_value=default_value, dtype=values.dtype
    ).masked_scatter_(to_mask, torch.masked_select(values, from_mask))
    return result


def min_value_of_mask(mask: torch.BoolTensor, dtype: torch.dtype) -> torch.tensor:
    result = torch.zeros_like(mask, dtype=dtype)
    result[~mask] = min_value_of_dtype(dtype)
    return result


def to_numpy(x: Union[numpy.ndarray, torch.Tensor]) -> numpy.ndarray:
    if isinstance(x, numpy.ndarray):
        return x
    return x.detach().cpu().numpy()


def argmax_onehot(x: torch.FloatTensor) -> torch.FloatTensor:
    _, feature_size = x.size()
    return torch.nn.functional.one_hot(
        torch.max(x, dim=-1).indices, num_classes=feature_size,
    ).float()


def prepend_dummy_zero_step(
    x: torch.Tensor,
    fill_value: Union[float, int, bool] = 1
) -> torch.Tensor:

    if len(x.size()) == 3:
        batch_size, _, hidden_dim = x.size()
        size = (batch_size, 1, hidden_dim)
    elif len(x.size()) == 2:
        batch_size, _ = x.size()
        size = (batch_size, 1)
    else:
        raise NotImplementedError(
            'this function supports only 2d or 3d tensor'
        )

    dummy = torch.full(
        size, fill_value=fill_value, dtype=x.dtype, device=x.device
    )
    return torch.cat([dummy, x], dim=1)


def cmb(n: int, r: int) -> int:
    if n < r:
        return 0
    r = min(n - r, r)
    if r == 0:
        return 1
    over = reduce(mul, range(n, n - r, -1))
    under = reduce(mul, range(1, r + 1))
    return over // under


def extend_token_type_ids_from_mask(
    orig_token_type_ids: torch.LongTensor,
    orig_input_mask: torch.BoolTensor,
    new_input_mask: torch.BoolTensor,
) -> torch.LongTensor:
    """construct a new token_type_ids which is compatible with
    the given input_mask tensor.

    Args:
        orig_token_type_ids (torch.LongTensor):
            (batch_size, sequence_length) token_type_ids tensor,
            whose values are 1 for at indices corresponding to
            tokens in passage text.
        orig_input_mask (torch.BoolTensor):
            (batch_size, sequence_length) original input mask tensor
        new_input_mask (torch.BoolTensor):
            (batch_size, new_sequence_length) input mask

    Returns:
        torch.LongTensor: token_type_ids compatible with `new_input_mask`
    """
    batch_size, seq_len1 = orig_token_type_ids.size()
    _, seq_len2 = new_input_mask.size()

    # assume the original question is preserved
    assert seq_len1 <= seq_len2
    # if seq_len1 > seq_len2:
    #     passage_mask = orig_token_type_ids[:, :seq_len2].cummax(
    #         dim=-1).values.bool()
    #     return (passage_mask & new_input_mask).long()

    if seq_len1 == seq_len2:
        return orig_token_type_ids

    # mask where the first [CLS], qustion_tokens, [SEP] are one
    # (note that passage tokens are assigned 2, paddings are zero)
    cls_question_sep_token_mask = (
        orig_token_type_ids + orig_input_mask.long()) == 1

    padded_cls_question_sep_token_mask = torch.cat(
        [
            cls_question_sep_token_mask,
            cls_question_sep_token_mask.new_zeros(
                batch_size, seq_len2 - seq_len1, dtype=torch.bool
            ),
        ],
        dim=1,
    )

    # now tokens in the passage text are one
    results = ~padded_cls_question_sep_token_mask & new_input_mask
    return results.long()


def get_token_type_ids_from_text_field_tensors(
    text_field_tensors: TextFieldTensors,
    sep_token_index: Optional[int] = None,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:

    for _, indexer_tensors in text_field_tensors.items():
        for argument_name, tensor in indexer_tensors.items():
            if argument_name == "type_ids":
                return tensor

    token_ids = get_token_ids_from_text_field_tensors(text_field_tensors)
    if mask is None:
        mask = get_text_field_mask(text_field_tensors)

    assert (
        len(token_ids.size()) == 2
        and len(mask.size()) == 2
        and sep_token_index is not None
        and mask is not None
    ), (
        "`type_ids` entry not found in the TextFieldTensors. "
        "`sep_token_index` and `mask` may need to be explicitly passed"
    )

    # find where the first [SEP] token is, and assign ones at indices after that.
    onehot_sep_tokens = (token_ids == sep_token_index).long()
    type_ids = (onehot_sep_tokens.cumsum(
        dim=-1) - onehot_sep_tokens > 0).long()
    type_ids = type_ids * mask.long()
    return type_ids


def find_valid_spans_in_arithmetic_hint(
    tokens: List[str],
    answer_numbers: List[Union[int, float]],
    include_more_numbers: bool = True,
) -> Iterator[Tuple[str, int, int]]:

    if len(tokens) == 0 or len(answer_numbers) == 0:
        return

    def _is_in_answer_numbers(number):
        return (
            number is not None
            and any(
                math.isclose(number, target)
                for target in answer_numbers
            )
        )

    def _to_number(texts):
        return convert_word_to_number(
            " ".join(texts).replace(" ##", "").replace(" ", ""),
            try_to_include_more_numbers=include_more_numbers
        )

    buf = []
    start, end = 0, 0
    for token in tokens:
        if token.startswith("##") or len(buf) and buf[-1] == "-":
            buf.append(token)
        else:
            concatenated = _to_number(buf)
            if _is_in_answer_numbers(concatenated):
                yield (concatenated, start, end - 1)
            buf = [token]
            start = end
        end += 1

    concatenated = _to_number(buf)
    if _is_in_answer_numbers(concatenated):
        yield (concatenated, start, end - 1)


def btag_representations_to_pair_argument_indices(
    btag_representations: Union[torch.LongTensor, torch.BoolTensor],
) -> torch.LongTensor:
    assert (
        btag_representations.dtype is torch.long
        or btag_representations.dtype is torch.bool
    )
    if btag_representations.dtype is torch.long:
        btag_representations = btag_representations > 0

    batch_size, seq_len = btag_representations.size()
    device = btag_representations.device
    candidate_mask = left_align_mask(
        btag_representations, reduce_size=True)

    argument_indices = mask_mask_copy(
        values=torch.arange(seq_len, device=device).unsqueeze(
            0).repeat(batch_size, 1),
        from_mask=btag_representations,
        to_mask=candidate_mask,
        default_value=0,
    )

    # shape (pair_argument_candidate_indices):
    #  (batch_size, max_#_argument_pairs, 2)
    # TODO: this is a bit ugly code
    results = torch.nn.utils.rnn.pad_sequence(
        [
            torch.combinations(candidate_indices[mask], 2)
            for candidate_indices, mask in zip(
                argument_indices, candidate_mask)
        ],
        batch_first=True,
    )
    return results


def mark_arguments_in_text_field_tensor_and_adjust_indices(
    text_field_tensor: TextFieldTensors,
    bio_tags_tensor: torch.LongTensor,
    argument_markers: List[List[str]],
    answer_as_passage_spans: torch.LongTensor,
    token_offsets_list: List[Tuple[int, int]],
    passage_tokens_list: List[List[str]],
    vocab: Vocabulary,
    namespace: str,
    max_position_index: int,
) -> Tuple[TextFieldTensors, torch.LongTensor, List[Optional[Tuple[int, int]]]]:

    NUM_ARGS = 2

    bio_tags_tensor = bio_tags_tensor.clone()
    bio_tags_tensor[:, 0] = 0  # erase tag on [CLS]
    num_tags = (bio_tags_tensor == BTAG).sum(dim=-1)

    assert len(argument_markers) == NUM_ARGS
    assert ((num_tags == 0) | (num_tags == NUM_ARGS)).all()

    if (num_tags == 0).all():
        return (
            text_field_tensor,
            answer_as_passage_spans,
            token_offsets_list,
            passage_tokens_list,
        )

    token_ids = get_token_ids_from_text_field_tensors(
        text_field_tensor
    )

    batch_size = token_ids.size(0)
    device = token_ids.device

    mask = get_text_field_mask(text_field_tensor)

    token_type_ids = get_token_type_ids_from_text_field_tensors(
        text_field_tensor,
        sep_token_index=vocab.get_token_index("[SEP]", namespace),
        mask=mask,
    )

    new_answer_as_passage_spans = answer_as_passage_spans.clone()

    new_passage_tokens_list = []
    new_token_offsets_list = []
    new_token_ids_list = []
    for batch_index, token_ids, bio_tags, token_offsets, passage_tokens in zip(
        range(batch_size),
        token_ids.cpu(),
        bio_tags_tensor.cpu(),
        token_offsets_list,
        passage_tokens_list,
    ):
        token_ids = token_ids.tolist()
        new_passage_tokens = []
        new_token_offsets = []
        new_token_ids = []
        prev_token_index = 0
        argument_index = 0
        iterator = enumerate(bio_tags)
        while True:
            token_index, bio_tag = next(iterator, (len(bio_tags), BTAG))
            if bio_tag == BTAG:
                new_answer_as_passage_spans[batch_index][
                    (prev_token_index <= answer_as_passage_spans[batch_index])
                    & (answer_as_passage_spans[batch_index] < token_index)
                ] += len(sum(argument_markers[:argument_index], []))

                new_token_ids += token_ids[prev_token_index:token_index]
                new_token_offsets += token_offsets[prev_token_index:token_index]
                new_passage_tokens += passage_tokens[prev_token_index:token_index]
                if argument_index < len(argument_markers):
                    new_token_ids += [
                        vocab.get_token_index(token, namespace)
                        for token in argument_markers[argument_index]
                    ]
                    new_token_offsets += [
                        (-1, -1) for _ in argument_markers[argument_index]
                    ]
                    new_passage_tokens += argument_markers[argument_index]

                prev_token_index = token_index
                argument_index += 1
            if token_index == len(bio_tags):
                break

        new_token_ids_list.append(new_token_ids)
        new_token_offsets_list.append(new_token_offsets)
        new_passage_tokens_list.append(new_passage_tokens)

    new_answer_as_passage_spans[
        new_answer_as_passage_spans.min(dim=-1).values >= max_position_index
    ] = -1

    tokens_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(new_token_ids, dtype=torch.long, device=device)
            for new_token_ids in new_token_ids_list],
        batch_first=True,
        padding_value=vocab.get_token_index("[PAD]", namespace),
    )

    padding = torch.zeros(
        batch_size,
        len(sum(argument_markers, [])),
        dtype=torch.bool,
        device=device
    )
    padding[num_tags == NUM_ARGS] = 1
    new_mask = left_align_mask(
        torch.cat([mask, padding], dim=-1),
        reduce_size=True
    )
    tokens_tensor = tokens_tensor[:, :new_mask.size(1)]

    new_type_ids = extend_token_type_ids_from_mask(
        orig_token_type_ids=token_type_ids,
        orig_input_mask=mask,
        new_input_mask=new_mask,
    )

    new_text_field_tensor = {
        'tokens': {
            'token_ids': tokens_tensor,
            'mask': new_mask,
            'type_ids': new_type_ids,
        }
    }

    return (
        new_text_field_tensor,
        new_answer_as_passage_spans,
        new_token_offsets_list,
        new_passage_tokens_list
    )


def copy_btag_values_to_itags(
    x: torch.Tensor,
    bio_tensor: torch.LongTensor,
    bio_output: bool = False,
) -> torch.Tensor:

    assert len(x.size()) in (2, 3)
    assert not bio_output or x.dtype is torch.long

    batch_size, seq_len = x.size()[:2]
    device = x.device

    indices = (
        torch.arange(seq_len, device=device).unsqueeze(
            0).repeat(batch_size, 1)
    )
    indices[bio_tensor == ITAG] = -1

    if len(x.size()) == 2:
        results = torch.gather(
            x, 1, indices.cummax(dim=-1).values
        )
    else:
        results = gather_row(
            x, indices.cummax(dim=-1).values
        )

    if bio_output:
        results += x

    return results


@ contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class VocabularyIndexAdapter(torch.nn.Module):
    def __init__(
        self, vocab: Vocabulary, from_namespace: str, to_namespace: str
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.from_namespace = from_namespace
        self.to_namespace = to_namespace

        from_token_indices, to_token_indices = [], []
        for index, token in sorted(
            self.vocab.get_index_to_token_vocabulary(self.to_namespace).items()
        ):
            if token in self.vocab.get_token_to_index_vocabulary(self.from_namespace):
                from_token_indices.append(
                    self.vocab.get_token_index(token, self.from_namespace)
                )
                to_token_indices.append(index)

        self._from_token_indices = torch.tensor(
            from_token_indices, dtype=torch.long)
        self._to_token_indices = torch.tensor(
            to_token_indices, dtype=torch.long)

    def shortest(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, from_vocab_size = x.size()
        assert from_vocab_size == self.vocab.get_vocab_size(
            self.from_namespace)

        from_token_indices = self._from_token_indices.to(device=x.device)
        to_token_indices = self._to_token_indices.to(device=x.device)

        results = x.new_zeros(
            batch_size, seq_len, self.vocab.get_vocab_size(self.to_namespace)
        ).index_copy_(2, to_token_indices, x[:, :, from_token_indices])

        return results


def test():
    embeddings = torch.arange(4 * 5 * 8).view(4, 5,
                                              8).float().requires_grad_(True)

    mask = torch.tensor(
        [[1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], ],
        dtype=torch.float,
    ).requires_grad_(True)

    padding = 1  # torch.arange(8).float()

    # import IPython; IPython.embed(colors='neutral')

    results, mask_ = masked_embedding_lookup_with_padding(
        embeddings, mask, padding)
    # import IPython; IPython.embed()
    results.sum().backward()
    print(results)
    print(mask_)
    print(embeddings.grad)
    print(mask.grad)


# test()
