# Run ledger

`runs.jsonl` (ruler and attack jobs) and `executor_runs.jsonl` (gap-map, moonshot and rung 0 jobs) list every GPU job in the order it was admitted and closed: job id, time, machine type, zone,
provisioning (spot), maximum run time and a note naming the run. It is the public copy of the experiment's
append-only job ledger. Cost fields and the hash chain over them are removed, matching the paper, which reports
total compute only ("a few hundred dollars"); project names are replaced by placeholders (gcp-project-a, -b, -c).

The pre-registration record (every bar, ruling and verdict, timestamped) is the critic log in
`governance/critic_log.md`.
