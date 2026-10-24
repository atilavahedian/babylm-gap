# GPT-BERT 10M vs 100M, dev sample 187,326 words / 287 pieces (reweighted to dev source shares) | words in verbatim train_100M lines dropped

nats/word: 10M 5.141, 100M 4.138, gap 1.003 (CI 0.895-1.154); raw gap 0.997

## Bands (train_10M count)
| band | word share | gap/word | gap share | gap nats/word overall |
|---|---|---|---|---|
| 0 | 1.845% | 7.95 | 14.6% | 0.147 |
| 1-63 | 6.699% | 2.90 | 19.4% | 0.194 |
| 64-8k | 35.517% | 1.21 | 42.7% | 0.429 |
| 8k+ | 55.939% | 0.42 | 23.3% | 0.234 |

## Never-seen words by type (share of band words / share of band gap / gap per word)
| type | share | gap share | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|
| non-ascii letters | 20.6% | 46.6% | 18.01 | 30.49 | 12.49 |
| capitalized novel word (name-like) | 25.8% | 17.2% | 5.28 | 19.92 | 14.64 |
| lowercase novel word | 11.7% | 8.5% | 5.77 | 22.46 | 16.69 |
| hyphen/slash compound of seen parts | 9.2% | 6.5% | 5.61 | 23.63 | 18.03 |
| inflection of seen word | 9.6% | 5.4% | 4.48 | 17.15 | 12.67 |
| closed compound of seen words | 6.9% | 4.2% | 4.81 | 18.87 | 14.06 |
| derivation of seen word | 7.9% | 3.8% | 3.80 | 14.84 | 11.04 |
| hyphen/slash, part unseen | 2.4% | 2.9% | 9.59 | 32.50 | 22.92 |
| symbols only | 1.8% | 2.8% | 12.40 | 15.12 | 2.71 |
| number/digits | 3.1% | 1.4% | 3.48 | 17.13 | 13.65 |
| two-affix derivation of seen word | 1.0% | 0.8% | 6.51 | 22.33 | 15.83 |

## Never-seen words by train_100M count
| c100 | share | gap share | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|
| 0 | 41.4% | 34.6% | 6.63 | 25.35 | 18.72 |
| 1-9 | 32.6% | 30.5% | 7.43 | 20.63 | 13.20 |
| 10-63 | 19.2% | 24.4% | 10.09 | 19.46 | 9.37 |
| 64+ | 6.8% | 10.6% | 12.36 | 17.50 | 5.14 |

## 1-63 band by train_10M count
| c10 | share | gap share | gap/word | nll10 | nll100 | median c100/c10 |
|---|---|---|---|---|---|---|
| 1 | 8.5% | 15.4% | 5.28 | 13.38 | 8.10 | 29.0 |
| 2-3 | 9.1% | 13.1% | 4.18 | 14.25 | 10.07 | 16.7 |
| 4-7 | 12.2% | 13.8% | 3.26 | 12.67 | 9.41 | 12.9 |
| 8-15 | 17.2% | 15.7% | 2.65 | 11.59 | 8.95 | 11.8 |
| 16-31 | 22.3% | 18.9% | 2.45 | 10.58 | 8.13 | 11.1 |
| 32-63 | 30.7% | 23.1% | 2.18 | 9.66 | 7.48 | 10.8 |

## 1-63 band by c100/c10 ratio
| ratio | share | gap share | gap/word |
|---|---|---|---|
| <3x | 1.6% | 1.0% | 1.85 |
| 3-7x | 10.4% | 7.2% | 2.02 |
| 7-15x | 56.0% | 46.4% | 2.40 |
| 15x+ | 32.0% | 45.3% | 4.10 |

## First use vs repeat (repeat = same normal form within the previous 300 words of the piece)
| band | split | share of band | gap share of band | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|---|
| 0 | first | 74.8% | 86.4% | 9.18 (6.05-12.66) | 26.19 | 17.02 |
| 0 | repeat | 25.2% | 13.6% | 4.30 (2.75-6.10) | 10.14 | 5.83 |
| 1-63 | first | 74.5% | 75.6% | 2.94 (2.82-3.06) | 12.63 | 9.69 |
| 1-63 | repeat | 25.5% | 24.4% | 2.77 (2.24-3.38) | 7.42 | 4.65 |
| 64-8k | first | 67.5% | 67.4% | 1.20 | 7.04 | 5.84 |
| 64-8k | repeat | 32.5% | 32.6% | 1.21 | 4.82 | 3.60 |
| 8k+ | first | 20.7% | 26.5% | 0.53 | 4.30 | 3.76 |
| 8k+ | repeat | 79.3% | 73.5% | 0.39 | 2.78 | 2.39 |

## First token vs spelling (continuation tokens), each model with its own tokenizer
| band | tok/word 10M | tok/word 100M | gap first | gap cont | first share | nll10 first/cont | nll100 first/cont |
|---|---|---|---|---|---|---|---|
| 0 | 4.26 | 3.94 | 1.30 | 6.65 | 16% | 6.95/15.19 | 5.66/8.54 |
| 1-63 | 2.87 | 2.75 | 1.40 | 1.50 | 48% | 7.43/3.87 | 6.03/2.37 |
| 64-8k | 1.45 | 1.45 | 1.03 | 0.17 | 86% | 5.40/0.92 | 4.37/0.74 |
| 8k+ | 1.35 | 1.35 | 0.37 | 0.04 | 89% | 2.64/0.45 | 2.27/0.41 |

## Rare words as context: frequent (64+) targets by distance to the last rare word
| rare def | distance | share of freq words | gap/word |
|---|---|---|---|
| rare<=63 | 1 | 8.0% | 1.01 |
| rare<=63 | 2 | 6.9% | 0.80 |
| rare<=63 | 3 | 6.1% | 0.77 |
| rare<=63 | 4-6 | 14.5% | 0.76 |
| rare<=63 | 7-15 | 24.9% | 0.70 |
| rare<=63 | 16-50 | 26.7% | 0.65 |
| rare<=63 | >50/none | 11.1% | 0.60 |
| rare<=63 | excess within 3 words, nats/word overall | 0.044 | CI 0.031-0.061 |
| never_seen | 1 | 1.5% | 1.08 |
| never_seen | 2 | 1.5% | 0.88 |
| never_seen | 3 | 1.4% | 0.84 |
| never_seen | 4-6 | 3.7% | 0.81 |
| never_seen | 7-15 | 8.8% | 0.79 |
| never_seen | 16-50 | 20.3% | 0.74 |
| never_seen | >50/none | 47.5% | 0.68 |
| never_seen | excess within 3 words, nats/word overall | 0.010 | CI 0.003-0.020 |

## 1-63 band by case
| case | share of band | gap share of band | gap/word |
|---|---|---|---|
| cap | 28.8% | 32.7% | 3.30 |
| lower | 71.2% | 67.3% | 2.74 |

## By source
| source | dev share | gap/word | gap share | rare word share | rare gap share (of total) |
|---|---|---|---|---|---|
| bnc_spoken | 12.1% | 0.60 | 7.3% | 4.5% | 1.3% |
| childes | 26.2% | 1.11 | 29.0% | 5.7% | 10.9% |
| gutenberg | 27.3% | 1.16 | 31.7% | 10.3% | 9.9% |
| open_subtitles | 20.1% | 0.71 | 14.2% | 7.7% | 3.9% |
| simple_wiki | 12.8% | 1.34 | 17.0% | 16.3% | 7.8% |
| switchboard | 1.4% | 0.54 | 0.8% | 3.7% | 0.1% |

## Top 1-63 words by gap mass

_[Section removed in the clean repository: per-word list of corpus words (BabyLM corpus licence; BNC/CHILDES/Switchboard terms). The original is at wip-branch:results/babylm/rare_words_v1/decomp_filtered.md.]_
