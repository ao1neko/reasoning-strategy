from typing import Optional
import torch

OTAG = 0
ITAG = 1
BTAG = 2
NUM_TAGS = 3
MAX_NUM_TAGS = 40
BTAG_FLAG = 1 << 41

tag_labels = {OTAG: "O", ITAG: "I", BTAG: "B"}


def _rand_int(high: torch.Tensor) -> torch.LongTensor:
    return (torch.rand(high.shape, device=high.device) * high).long()


def num_tags_in_tensor(flags: torch.LongTensor) -> torch.LongTensor:
    assert len(flags.size()) == 2 and max(flags[:, 0]) <= MAX_NUM_TAGS
    return flags[:, 0]


def _sample_tags(
    log_probs: torch.FloatTensor, flags: torch.LongTensor, argmax_sampling: bool = True,
) -> torch.LongTensor:
    num_tags = num_tags_in_tensor(flags).cpu()
    flags_ = flags.clone()
    flags_[~(flags_ & BTAG_FLAG).bool()] = 0
    flags_ = (flags_ & ~BTAG_FLAG).cpu()
    batch_size, _ = flags_.shape

    sampled_indices = torch.zeros(batch_size, dtype=torch.long)
    for batch_index in range(batch_size):
        if num_tags[batch_index] == 0:
            continue

        logits = torch.zeros(num_tags[batch_index], dtype=torch.float)
        for tag_index in range(num_tags[batch_index]):
            log_prob = sum(
                log_probs[batch_index, token_index]
                for token_index, flag in enumerate(flags_[batch_index])
                if flag & (1 << tag_index)
            )
            logits[tag_index] = log_prob

        if argmax_sampling:
            sampled_indices[batch_index] = torch.argmax(logits, dim=0)
        else:
            sampled_index = torch.distributions.Categorical(
                logits=logits).sample()
            sampled_indices[batch_index] = sampled_index

    sampled_indices = sampled_indices.to(flags.device)

    return sampled_indices


def sample_tags(
    flags: torch.LongTensor,
    *,
    log_probs: Optional[torch.FloatTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    bio: bool = True,
    argmax_sampling: bool = True,
) -> torch.LongTensor:
    assert len(flags.size()) == 2

    assert log_probs is None or indices is None

    if indices is None:
        num_tags = num_tags_in_tensor(flags)
        indices = _rand_int(num_tags)
    elif log_probs is None:
        indices = _sample_tags(
            log_probs, flags, argmax_sampling=argmax_sampling)
    else:
        assert len(indices.size()) == 1 and indices.size(0) == flags.size(0)

    indices = (torch.ones_like(indices) << indices).unsqueeze(dim=-1)

    tags = flags & ~BTAG_FLAG
    tags[:, 0] = OTAG
    sampled_tags = (tags & indices).bool().long()

    if not bio:
        return sampled_tags

    btags = flags.clone()
    btags[:, 0] = OTAG
    sampled_btags = (btags & (indices | BTAG_FLAG)
                     == (indices | BTAG_FLAG)).long()
    return sampled_btags + sampled_tags


def sample_all_tags(flags: torch.LongTensor) -> torch.LongTensor:
    assert len(flags.size()) == 2

    tags = (flags >= BTAG_FLAG).long()
    return tags


def all_tag_indices(flags: torch.LongTensor) -> torch.LongTensor:
    assert len(flags.size()) == 2
    num_tags = num_tags_in_tensor(flags)
    batch_size, seq_len = flags.size()
    results = torch.full(
        (batch_size, num_tags.max().item(), 2),
        fill_value=-1,
        dtype=torch.long
    )

    flags_ = flags.clone().cpu()
    flags_[:, 0] = 0
    for batch_index in range(batch_size):
        for tag_index in range(num_tags[batch_index]):
            flag = BTAG_FLAG | (1 << tag_index)
            results[batch_index, tag_index] = torch.arange(
                seq_len)[(flags_[batch_index] & flag) == flag]

    return results


def mark_all_tags_in_candidates(
    flags: torch.LongTensor,
    candidate_indices: torch.LongTensor
) -> torch.BoolTensor:

    device = flags.device
    assert (
        len(flags.size()) == 2 and len(candidate_indices.size()) == 3
    )

    flags = flags.cpu()
    candidate_indices = candidate_indices.cpu()

    num_tags = num_tags_in_tensor(flags)
    batch_size, seq_len = flags.size()
    num_candidates = candidate_indices.size(1)

    results = torch.zeros(batch_size, num_candidates, dtype=torch.bool)
    numbers = torch.arange(seq_len)

    flags_ = flags.clone().cpu()
    flags_[:, 0] = 0
    for batch_index in range(batch_size):
        for tag_index in range(num_tags[batch_index]):
            flag = BTAG_FLAG | (1 << tag_index)
            candidate_in_this_index = \
                numbers[flags_[batch_index] & flag == flag]
            results[batch_index] |= torch.all(
                candidate_indices[batch_index] == candidate_in_this_index, dim=1)

    results = results.to(device)

    return results
