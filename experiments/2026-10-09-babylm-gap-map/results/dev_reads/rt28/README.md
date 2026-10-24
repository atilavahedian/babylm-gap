# 28L retrain reads (executor 3; VM bl-ms-28l-retrain-zs-1011-093615, project-c, job ms_28l_retrain_zs.json @ref-0348)
Retrained members: solo 2.9.1, keep_checkpoints true; twin dumps ref-0034 (--check vs own twin json passed on VM). calib_set as reads/k2_estar.

| member | retrain twin full / guarded | dev-read twin full | diff |
|---|---|---|---|
| e5 s0 rt | 4.9209 / 4.8953 | 4.9203 | +0.0006 |
| e5 s1 rt | 4.9297 / 4.9036 | 4.9325 | -0.0028 |
| e6 s0 rt | 4.9135 / 4.8902 | 4.9158 | -0.0023 |
| e5 s2 rt | 4.9293 / 4.9047 | 4.9319 | -0.0026 |
| e5 s3 rt | 4.9296 / 4.8989 | 4.9253 | +0.0043 |
| e6 s1 rt | 4.9211 / 4.8933 | 4.9206 | +0.0005 |
| e5 s4 rt | 4.9288 / 4.9031 | 4.9244 | +0.0044 |
| e5 s5 rt | 4.9192 / 4.8924 | 4.9203 | -0.0011 |
| e6 s2 rt | 4.9174 / 4.8933 | 4.9221 | -0.0047 |

## Pair p01 (e5 s0 + s1), 2026-10-11 ~11:00Z
mix raw 4.793700 / 4.770492 (MIX_REF_NPW full raw = 4.793700354253721); cal_t 4.7866 / 4.7640 (T 1.05). Dev-read M28a cal_t 4.7873 / 4.7640.

## Pair p23 (e5 s2 + s3), 2026-10-11 ~12:40Z
mix raw 4.795396 / 4.771589 (MIX_REF_NPW full raw = 4.795395544528819); cal_t 4.7884 / 4.7652 (T 1.05). Dev-read M28b cal_t 4.7896 / 4.7653.

## Pair p45 (e5 s4 + s5), 2026-10-11 ~14:50Z
mix raw 4.792486 / 4.768748 (MIX_REF_NPW full raw = 4.792486397887453); cal_t 4.7851 / 4.7620 (T 1.05). Dev-read M28c cal_t 4.7835 / 4.7615.

## Retrained M28 gain (all 9 retrained members, one calib_set; 2026-10-11 ~15:20Z)
Reference = mean of retrained e6 s0-s2: raw 4.9173 / 4.8922, cal_t 4.8991 / 4.8749.
| mixture | raw gain full / guarded | cal_t gain full / guarded |
|---|---|---|
| M28rt_p01 | 0.1236 / 0.1218 | 0.1125 / 0.1110 |
| M28rt_p23 | 0.1219 / 0.1207 | 0.1107 / 0.1097 |
| M28rt_p45 | 0.1248 / 0.1235 | 0.1139 / 0.1130 |
| mean | 0.1235 / 0.1220 | 0.1124 / 0.1112 |
Dev-read A3 (reads/ls_28l/rep3): cal_t mean +0.1145 / +0.1116 (raw +0.1257 / +0.1224). The retrained set reproduces it within 0.002 cal_t.
