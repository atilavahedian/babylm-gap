# A5 T100: port recipe at E10, seeds 1 and 2 (A100, twin ref-0026 + dump ref-0034, code ref-0165, torch 2.7.0)

Full DEV raw, nats/word (twin):
| seed | twin | dump check |
|---|---|---|
| s0 (grid_S_100M_e10_s0, GPU twin) | 4.2612 | A100 temps dump pending (retry on x0) |
| s1 | 4.25636 | 4.25636 |
| s2 | 4.25783 | 4.25783 |

- Mean over s0-s2 is 4.2585 raw (seed SD ~0.0025).
- Each seed's temps npz (~234 MB) is in GCS only: ${BUCKET}/babylm/vms/bl-a5-100m-port-s1s2-1011-232508/out/babylm/runs/a5port_S_100M_e10_s{1,2}/pt_v2/dev_v2.npz.

## Calibrated T100 (executor 3, 2026-10-12 10:45Z)
calib_set K=1 (sha256 ref-0374…, CALIB_TOKENIZER = base_S_e6 export tokenizer) over each seed's full-DEV temps dump:
s0 = x0 redump on A100 (bl-final-ii-28l-x0-1012-053724, out/a5port_s0_redump; twin ref-0026, dump ref-0034, dev_bpw ref-0165 sha ref-0195), s1/s2 = port s1s2 VM dumps. File: calib_port_s0s2.json.

| seed | raw full | cal_t full | raw guarded | cal_t guarded | t |
|---|---|---|---|---|---|
| s0 | 4.26116 | 4.24037 | 4.25724 | 4.23649 | 1.1 |
| s1 | 4.25636 | 4.23798 | 4.25259 | 4.23423 | 1.05 |
| s2 | 4.25783 | 4.23695 | 4.25411 | 4.23324 | 1.1 |
| mean | 4.25845 | 4.23843 | 4.25465 | 4.23465 | |
| sd | 0.00246 | 0.00176 | 0.00237 | 0.00166 | |
