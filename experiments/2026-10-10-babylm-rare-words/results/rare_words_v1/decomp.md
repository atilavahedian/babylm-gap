# GPT-BERT 10M vs 100M, dev sample 215,312 words / 287 pieces (reweighted to dev source shares)

nats/word: 10M 5.031, 100M 4.060, gap 0.971 (CI 0.874-1.094); raw gap 0.965

## Bands (train_10M count)
| band | word share | gap/word | gap share | gap nats/word overall |
|---|---|---|---|---|
| 0 | 1.709% | 7.68 | 13.5% | 0.131 |
| 1-63 | 6.409% | 2.92 | 19.3% | 0.187 |
| 64-8k | 34.872% | 1.20 | 43.3% | 0.420 |
| 8k+ | 57.010% | 0.41 | 24.0% | 0.233 |

## Never-seen words by type (share of band words / share of band gap / gap per word)
| type | share | gap share | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|
| non-ascii letters | 18.4% | 41.7% | 17.39 | 29.63 | 12.25 |
| capitalized novel word (name-like) | 27.0% | 19.1% | 5.44 | 19.98 | 14.54 |
| lowercase novel word | 11.5% | 8.9% | 5.92 | 22.64 | 16.72 |
| hyphen/slash compound of seen parts | 9.6% | 7.2% | 5.72 | 23.60 | 17.88 |
| inflection of seen word | 9.5% | 5.7% | 4.64 | 17.33 | 12.69 |
| closed compound of seen words | 6.9% | 4.4% | 4.85 | 19.02 | 14.17 |
| derivation of seen word | 8.0% | 4.1% | 3.92 | 15.22 | 11.29 |
| hyphen/slash, part unseen | 2.6% | 3.4% | 10.09 | 32.36 | 22.27 |
| symbols only | 2.0% | 3.1% | 12.01 | 15.19 | 3.18 |
| number/digits | 3.5% | 1.6% | 3.45 | 16.99 | 13.53 |
| two-affix derivation of seen word | 1.0% | 0.9% | 6.47 | 22.43 | 15.95 |

## Never-seen words by train_100M count
| c100 | share | gap share | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|
| 0 | 39.7% | 30.4% | 5.88 | 24.80 | 18.92 |
| 1-9 | 33.3% | 30.8% | 7.09 | 20.53 | 13.44 |
| 10-63 | 19.5% | 25.3% | 9.95 | 19.32 | 9.36 |
| 64+ | 7.4% | 13.4% | 13.90 | 18.75 | 4.85 |

## 1-63 band by train_10M count
| c10 | share | gap share | gap/word | nll10 | nll100 | median c100/c10 |
|---|---|---|---|---|---|---|
| 1 | 8.1% | 14.3% | 5.16 | 13.51 | 8.35 | 34.0 |
| 2-3 | 9.0% | 13.2% | 4.30 | 14.41 | 10.10 | 17.0 |
| 4-7 | 12.4% | 14.0% | 3.31 | 12.55 | 9.24 | 13.2 |
| 8-15 | 17.4% | 16.2% | 2.73 | 11.63 | 8.90 | 11.9 |
| 16-31 | 22.5% | 19.2% | 2.49 | 10.52 | 8.03 | 11.1 |
| 32-63 | 30.7% | 23.0% | 2.18 | 9.65 | 7.47 | 10.8 |

## 1-63 band by c100/c10 ratio
| ratio | share | gap share | gap/word |
|---|---|---|---|
| <3x | 1.7% | 1.1% | 1.94 |
| 3-7x | 10.6% | 7.7% | 2.13 |
| 7-15x | 56.0% | 46.5% | 2.42 |
| 15x+ | 31.7% | 44.6% | 4.11 |

## First use vs repeat (repeat = same normal form within the previous 300 words of the piece)
| band | split | share of band | gap share of band | gap/word | nll10 | nll100 |
|---|---|---|---|---|---|---|
| 0 | first | 74.7% | 84.3% | 8.66 (6.16-11.56) | 25.70 | 17.04 |
| 0 | repeat | 25.3% | 15.7% | 4.77 (2.88-6.79) | 10.52 | 5.75 |
| 1-63 | first | 74.9% | 76.2% | 2.97 (2.85-3.09) | 12.57 | 9.61 |
| 1-63 | repeat | 25.1% | 23.8% | 2.76 (2.32-3.31) | 7.45 | 4.69 |
| 64-8k | first | 67.3% | 67.1% | 1.20 | 7.02 | 5.82 |
| 64-8k | repeat | 32.7% | 32.9% | 1.21 | 4.81 | 3.60 |
| 8k+ | first | 20.1% | 25.8% | 0.53 | 4.30 | 3.77 |
| 8k+ | repeat | 79.9% | 74.2% | 0.38 | 2.74 | 2.36 |

## First token vs spelling (continuation tokens), each model with its own tokenizer
| band | tok/word 10M | tok/word 100M | gap first | gap cont | first share | nll10 first/cont | nll100 first/cont |
|---|---|---|---|---|---|---|---|
| 0 | 4.27 | 3.97 | 1.28 | 6.39 | 17% | 6.91/14.94 | 5.63/8.55 |
| 1-63 | 2.90 | 2.77 | 1.39 | 1.53 | 48% | 7.36/3.93 | 5.97/2.40 |
| 64-8k | 1.51 | 1.51 | 1.01 | 0.19 | 84% | 5.27/1.03 | 4.26/0.83 |
| 8k+ | 1.45 | 1.45 | 0.36 | 0.05 | 87% | 2.52/0.54 | 2.16/0.48 |

## Rare words as context: frequent (64+) targets by distance to the last rare word
| rare def | distance | share of freq words | gap/word |
|---|---|---|---|
| rare<=63 | 1 | 7.6% | 1.02 |
| rare<=63 | 2 | 6.6% | 0.81 |
| rare<=63 | 3 | 5.8% | 0.79 |
| rare<=63 | 4-6 | 13.8% | 0.76 |
| rare<=63 | 7-15 | 24.0% | 0.71 |
| rare<=63 | 16-50 | 27.1% | 0.63 |
| rare<=63 | >50/none | 13.4% | 0.56 |
| rare<=63 | excess within 3 words, nats/word overall | 0.053 | CI 0.040-0.070 |
| never_seen | 1 | 1.4% | 1.07 |
| never_seen | 2 | 1.4% | 0.88 |
| never_seen | 3 | 1.3% | 0.87 |
| never_seen | 4-6 | 3.5% | 0.81 |
| never_seen | 7-15 | 8.4% | 0.81 |
| never_seen | 16-50 | 19.4% | 0.76 |
| never_seen | >50/none | 49.3% | 0.66 |
| never_seen | excess within 3 words, nats/word overall | 0.010 | CI 0.005-0.019 |

## 1-63 band by case
| case | share of band | gap share of band | gap/word |
|---|---|---|---|
| cap | 29.2% | 33.6% | 3.36 |
| lower | 70.8% | 66.4% | 2.74 |

## By source
| source | dev share | gap/word | gap share | rare word share | rare gap share (of total) |
|---|---|---|---|---|---|
| bnc_spoken | 12.1% | 0.60 | 7.5% | 4.4% | 1.3% |
| childes | 26.2% | 1.00 | 27.1% | 4.8% | 9.4% |
| gutenberg | 27.3% | 1.17 | 33.0% | 10.4% | 10.3% |
| open_subtitles | 20.1% | 0.67 | 14.0% | 6.8% | 3.6% |
| simple_wiki | 12.8% | 1.35 | 17.7% | 16.3% | 8.1% |
| switchboard | 1.4% | 0.50 | 0.7% | 3.3% | 0.1% |

## Top 1-63 words by gap mass

_[Section removed in the clean repository: per-word list of corpus words (BabyLM corpus licence; BNC/CHILDES/Switchboard terms). The original is at wip-branch:results/babylm/rare_words_v1/decomp.md.]_
