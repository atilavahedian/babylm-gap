# Reference labels (critic ref-0386 correction, executor 3 20:35Z)
The read names come from v2_readouts.sh slots, not from what the refs are:
- ls_K2_vs_e10: ref = solo E10 pool base_r1 ve0_tied s0-s4, torch 2.9.1 (cross-env vs the 2.7 members). Not an E6 read.
- ls_K2_vs_estar: ref = the two e5 members themselves (v2 e5 s0, s1; torch 2.7), i.e. mixture vs its own singles = g(2). NOT vs E6.
A true vs-E6-pool read (E6 s0-s4, 2.7, cal_t both legs) comes from reads/k2_estar/ (calib_set).
