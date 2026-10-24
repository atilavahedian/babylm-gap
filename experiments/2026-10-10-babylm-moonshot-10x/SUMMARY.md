# BabyLM 10x moonshot: portfolio, ceiling map and the F-cards see `../2026-10-09-babylm-gap-map/SUMMARY.md`)

## Question
Under BabyLM strict-small R0 (10M unique words, <= 10 epochs, <= 100M words of exposure counting every model,
teachers and generators included), does any mechanism have a ceiling that could carry a 10x data-efficiency gain, and
which channel (targets, contexts, posterior averaging) could carry it?

## Method
- **Portfolio.** Three design rounds (W1 and W2 on Opus with 9 lenses, W3 on design agent 5.1 with 3 whole-system lenses),
  92 agents; generators, triage, three adversarial skeptics (prior art, rules, mechanism); 18 surviving cards with
  judged ceilings, expected gains and P(pass). Judgements, not measurements.
- **Ceiling map (diagnostic, out of track; critic ref-0126).** S base, 10M x 10 epochs, full DEV through per-token
  dumps of the registered scorer. C3: K-seed probability averages (K = 2..8) give g(K) and g_inf. C1: KD students on
  the 10M contexts with a 10M teacher (C1-self) or the 100M teacher T100 (C1-oracle); f1 = (C1-self - C1-oracle) /
  (C1-self - T100). Registered bands: f1 in [0.30, 0.75], g(8) in [0.10, 0.25].
- **Card PE** (one-forward shared-trunk particle ensemble; W1 rank 1). Gate 2 on a 1M mini-ruler proxy: kill if
  rho = J(PE3) / J(I3) < 0.5 (share of three independent models' gain kept).
- **Card F1** (8-model ensemble distilled into one student; design agent escalation). Stage 1 on the proxy: kill if the
  student beats the binding comparator by less than 2 floor = 0.107, or phi_stu < 0.20.
- **Card F2** (heterogeneous K=2 split; members of different kinds). Stage 0 on existing dumps, same-env pairs: dead
  unless d = g_cal(hetero) - g_cal(seed) >= 0.03 on both legs and the mixture beats the seed mixture by >= 0.016.
- **Legal-split family** (registered 2026-10-10T07:19Z): K=2 (2 x 5 epochs), K=3 (3 x 3.33), trunk (4 epochs) + 2
  branches (3 epochs each). R4.4 menus of single models at matched FLOPs: F=2 (S) and F=4.146 (28L, 7 cells).
- **Registered predictions** (design agent memo, `docs/fable2_legal_ceiling.md`): P-F2a..e, scored once on cal_t.

## Results
| Item | Measured | Bar / registered band | Verdict |
|---|---|---|---|
| Portfolio | every card's ceiling 0.05-0.30 nats/word, expected 0.00-0.03; union ~0.40-0.45 (~3x), P(all) ~1e-5 | 10x-sized ceiling | none found |
| C3 posterior channel | g(8) 0.2893 / 0.2815; g_inf 0.3327 / 0.3237 (raw, 800M exposure) | g(8) in [0.10, 0.25] | above band; diagnostic only |
| C1 targets channel | f1 0.697 raw / 0.704 overlap-removed (full); 0.714 / 0.721 (guarded) | [0.30, 0.75] | prediction holds (critic, row 117) |
| PE Gate 2 (b) | rho 0.364 / 0.365 | >= 0.5 | KILLED (ref-0325) |
| F1 stage 1 | student vs best comparator +0.0502 / +0.0506 | 0.107 | KILLED (critic entry 116) |
| F2 Stage 0 lane A (A2a input-noise member) | d +0.0059 / +0.0046 and +0.0068 / +0.0059; mixture gain <= 0.0102 | d >= 0.03 and >= 0.016 | KILLED |
| F2 Stage 0 lane B (a3a BPE-dropout member) | vs homogeneous noisy split +0.0048 / +0.0050 and +0.0032 / +0.0034 | 0.016 | KILLED (card verdict 21:15Z) |
| Trunk + 2 branches | -0.0373 / -0.0373 vs plain K=2 (cross-env; abs value > ENV_SYS 0.014) | - | dominated by plain K=2 |
| K=3 split | first reads scored mid-anneal (chck_30M) | - | VOID, no valid number |
| P-F2d: K=2 vs best F=2 single | +0.058 / +0.059 cal_t | [0.05, 0.08] | HIT (critic ref-0186) |
| P-F2e: K=2 mixture minus size-M best (wd1.6 e3) | -0.330 | [-0.06, +0.02] | MISS (M is 0.14-0.20 worse than the S single) |
| F=4.146 menu (28L comparator) | best 36L/512d e4 4.8728 / 4.8461; 28L split 4.7868 / 4.7636: +0.086 / +0.083 | 0.012 | split PASSES (critic ref-0182) |

## What it rules out
- A 10x mechanism among the 18 surviving known-idea cards: no card's judged ceiling exceeds 0.30 nats/word.
- One-forward or shared-trunk ensembles as a legal way to collect the posterior variance: sharing the trunk, head and
  data order kept 36% of independent models' gain (PE) and the trunk-plus-branches split lost 0.037 to plain K=2.
- Distilling an ensemble into one student: the student keeps its own ~0.29 of variance, so 8 teachers beat 1 by only 0.05 (proxy).
- Regulariser heterogeneity as a diversity source: input-noise and BPE-dropout members mix like another seed.
- "The split is parameter count": it beats every compute-matched single at both sizes.

## What it does not rule out / caveats
- Portfolio ceilings and odds are agent judgements, not measurements; only PE, F1 and F2 were run.
- PE and F1 were read on a 1M mini-ruler proxy (labels and kills only, never wins); a proxy kill does not price the
  same design at 10M words.
- F2 never read architecture- or objective-heterogeneous members (28L/384d, 12L/592d, size M, GPT-BERT): their runs
  were in another torch env (lane C, void until v2e). Gontijo-Lopes et al. find those, not regularisers, decorrelate errors.
- C1/C3 use teachers or ensembles beyond the 100M cap: they describe the data, not a legal route. The D1 bias map that
  followed (`../2026-10-09-babylm-gap-map/results/d1_bias/`) puts the perfect-legal-learner ceiling at <= 0.45 of a
  0.743 gap (~4x on that map's metric); earlier drafts quoted ~3x on the retired 0.83 yardstick
  (`../2026-10-09-babylm-gap-map/docs/FINAL_REPORT_DRAFT.md`, section 1).
- K=3 has no valid read; the branch read crosses torch envs (2.9.1 branch vs 2.7.0 K=2).
- F1 stage-1 per-run twin jsons (`babylm/moonshot/f1/`) cited by the critic are not in any git branch; the
  numbers above are the critic's recomputation (critic log row 116).

## References
- [docs/portfolio_v1.md](docs/portfolio_v1.md), [docs/ceiling_map.md](docs/ceiling_map.md), [docs/context_v1.md](docs/context_v1.md), [docs/fable2_legal_ceiling.md](docs/fable2_legal_ceiling.md).
- Cards: [docs/cards/F1_closed_system_bayes_brute_force.md](docs/cards/F1_closed_system_bayes_brute_force.md), [docs/cards/F2_heterogeneous_legal_split.md](docs/cards/F2_heterogeneous_legal_split.md), PE = [docs/cards/W1_one-forward-shared-trunk-particle-ensemble-submitted-as-one-.md](docs/cards/W1_one-forward-shared-trunk-particle-ensemble-submitted-as-one-.md); other W cards in [docs/cards/](docs/cards/); design-round inputs in [docs/refs/](docs/refs/).
- Reads: [results/ceiling_v2/summary.txt](results/ceiling_v2/summary.txt), [results/pe_gate2/summary.txt](results/pe_gate2/summary.txt), [results/f2_stage0/README.md](results/f2_stage0/README.md), [results/f4_menu/README.md](results/f4_menu/README.md), [results/ceiling_v2/ls_br/ENV_SYS_READ.md](results/ceiling_v2/ls_br/ENV_SYS_READ.md).
- Rulings: [governance/critic_log_moonshot.md](governance/critic_log_moonshot.md); full log in `../2026-10-09-babylm-gap-map/governance/critic_log.md`.
