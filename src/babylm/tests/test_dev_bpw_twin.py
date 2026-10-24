# Moved from tests/ (o3stxz) to babylm/tests/ in the clean tree; the pinned-list path below was adjusted for the move.
"""Checks for the blank-line twin of the DEV metric (no checkpoints or data needed)."""
from babylm.eval.dev_bpw_twin import TwinTokenizer, documents


class CharTok:
    """Stand-in tokenizer: one id per character, BOS 0, no special tokens added."""
    bos_token_id, eos_token_id = 0, 0

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {'input_ids': [ord(c) for c in text]}


def test_documents_split_at_blank_line_runs():
    assert documents('A\nB\n') == ['A\nB\n']
    assert documents('A\n\nB\n') == ['A', 'B\n']
    assert documents('\n \nA b\n \n\t\n\nB\nC\n\n\n') == ['A b', 'B\nC\n']
    assert documents('A\n\n  ') == ['A\n']
    assert documents('A') == ['A']


def test_twin_ids_equal_registered_without_blank_lines():
    tok, twin = CharTok(), TwinTokenizer(CharTok())
    text = 'a line\nanother  line\n'
    assert twin(text)['input_ids'] == tok(text)['input_ids']
    two = twin('one\ntwo\n\nthree\n')['input_ids']
    assert two == tok('one\ntwo')['input_ids'] + [0] + tok('three\n')['input_ids']
    assert twin.separators == [0, 1] and twin.bos_token_id == 0


class FastCharTok(CharTok):
    """CharTok with offsets, as a fast tokenizer gives them."""
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = super().__call__(text, add_special_tokens)
        if return_offsets_mapping:
            out['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
        return out


def test_unseen_chars_are_the_pinned_list():
    import json
    from pathlib import Path
    from babylm.eval.dev_bpw_twin import UNSEEN_CHARS
    pinned = json.loads((Path(__file__).parents[1] / 'tools/unseen_chars_list.json').read_text(encoding='utf-8'))
    assert UNSEEN_CHARS == ''.join(c['char'] for c in pinned['chars']) and len(UNSEEN_CHARS) == pinned['count'] == 85


def test_unseen_mask_drops_whole_words_and_keeps_whitespace():
    from babylm.eval.dev_bpw_twin import unseen_mask
    text = 'ok ⌈yes⌉ no\n*CHI: ↗'
    mask, words = unseen_mask(text, [(0, 2), (2, 4), (4, 8), (8, 9), (9, 11), (11, 12), (12, 17), (17, 19)])
    assert words == 2 and mask.tolist() == [False, True, True, False, False, False, False, True]


def test_twin_marks_unseen_tokens_and_never_the_separator():
    twin = TwinTokenizer(FastCharTok())
    ids = twin('a ⌈b\n\n⌉ c\n')['input_ids']
    mask, words = twin.unseen[0]
    assert len(mask) == len(ids) and words == 2
    assert ''.join('x' if m else ('|' if i == 0 else chr(i)) for i, m in zip(ids, mask)) == 'a xx|x c\n'


def test_nll_tokens_total_is_the_registered_total():
    import pytest
    torch = pytest.importorskip('torch')
    R = pytest.importorskip('babylm.eval.dev_bpw')  # the registered scorer, when it sits beside the twin
    from babylm.eval.dev_bpw_twin import nll_tokens

    class Tiny(torch.nn.Module):  # causal: running mean of embeddings -> logits
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.e, self.o = torch.nn.Embedding(40, 8), torch.nn.Linear(8, 40)

        def forward(self, input_ids, attention_mask):
            h = self.e(input_ids).cumsum(1) / torch.arange(1, input_ids.shape[1] + 1)[None, :, None]
            return {'logits': self.o(h)}

    ids = torch.randint(1, 40, (157,), generator=torch.Generator().manual_seed(1)).tolist()
    for kw in ({}, {'vocab_mask': 36}) if 'vocab_mask' in R.nll_of.__code__.co_varnames else ({},):
        total, tn = nll_tokens(Tiny(), ids, 0, 'cpu', bs=3, CTX=16, STRIDE=8, **kw)
        assert total == R.nll_of(Tiny(), ids, 0, 'cpu', bs=3, CTX=16, STRIDE=8, **kw)
        assert len(tn) == len(ids) and abs(tn.sum() - total) < 1e-6 * abs(total)  # fp32 job sums
