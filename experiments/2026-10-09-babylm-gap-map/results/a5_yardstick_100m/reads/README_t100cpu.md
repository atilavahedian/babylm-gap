# A5 yardstick on the untuned T100 (CPU dump): CPU/GPU mixed, untuned, diagnostic only

- T100 = grid_S_100M_e10_s0, which uses the port recipe untuned at 100M. It was dumped on CPU (v2 twin + temps).
  - full raw 4.26116, guarded raw 4.25724. These match the GPU references 4.2612 / 4.2572 within 1e-4.
  - cal_t full 4.24037 (t=1.1).
- Pool = S E6 s0..s4 dumps. Tool: babylm/moonshot/a5_yardstick.py @ref-0384.
- Y (all DEV): full raw 0.708 / cal_t 0.713; guarded raw 0.683 / cal_t 0.689. With overlap removed (ovr), every value is within 0.006 of these.
- x at cal_t full: S F2 1.21, 28L F4 1.32, S same-size 1.44, 28L same-size 1.45.
- Label: "x upper bound (untuned 100M)". A5 voided this point as the yardstick; the tuned 100M grid decides it.
- Critic (20:56Z): the CPU T100 dump is read against GPU pool dumps. That mix breaks the same-instrument rule (critic 08:54Z). Label: "CPU/GPU mixed, untuned". It is not a yardstick. The tuned grid uses A100 full-DEV temps dumps plus dev_bpw_twin.json, the same instrument as the S E6 pool.
