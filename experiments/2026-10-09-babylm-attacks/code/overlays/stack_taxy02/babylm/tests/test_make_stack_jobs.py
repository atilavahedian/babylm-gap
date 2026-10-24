"""make_stack_jobs: every stack job is the winner plus attack flags, a seed and a tag (docs/babylm/attacks/STACKING.md)."""
import json

import pytest

from babylm.configs import make_stack_jobs as M

A1 = "babylm/configs/jobs_a1_r1_S.json:r1_A1b"
A2 = "babylm/configs/jobs_a2_screen.json:a2_A2b"
A3 = "babylm/configs/jobs_a3_screen_S_r1win_wd0p4_ve0_tied.json:a3_S_r1win_wd0p4_ve0_tied_s0_a3a_all_p0p1_anneal"
A4 = "babylm/configs/jobs_a4_screen.json:a4_a"
SPECS = {"A1": A1, "A2": A2, "A3": A3, "A4": A4}
STACKS = [("A1", "A2"), ("A1", "A3"), ("A1", "A2", "A3"), ("A1", "A2", "A4"), ("A2", "A4")]


def arms_in(name, all_arms):
    label = name.rsplit("_s", 1)[0][len("stack_"):]
    return [] if label == "base" else [k for k in all_arms if k.lower() in label]


@pytest.mark.parametrize("arms", STACKS, ids=lambda a: "+".join(a))
def test_jobs_differ_from_the_winner_only_in_attack_flags_seed_and_tag(arms):
    spec = M.build({k: SPECS[k] for k in arms}, [0, 1, 2], loo=True)
    w = M.load_job(M.WINNER)
    wns = M.parsed(w["args"])
    arm_ns = {k: M.parsed(M.load_job(SPECS[k])["args"]) for k in arms}
    pad = 16512 if "A2" in arms else None
    n = len(arms)
    # arms whose own runs are unpadded get a padded single-arm comparator when the stack is padded (critic ref-0021)
    extra = [k for k in arms if pad and n > 2 and k != "A2"]
    assert len(spec["jobs"]) == 3 * (1 + n + len(extra)) + (3 if pad else 0)
    assert len({j["name"] for j in spec["jobs"]}) == len(spec["jobs"])
    assert "Pool: promotion" in spec["note"] and spec["pool"] == "promotion"
    assert spec["solo"] is True and "SOLO" in spec["note"]  # claims never run packed (critic ref-0001)
    w_rest = [g for g in M.groups(w["args"]) if g[0] not in ("--tag", "--seed")]
    for j in spec["jobs"]:
        used = arms_in(j["name"], arms)
        assert len(used) in (0, n, n - 1) or (len(used) == 1 and used[0] in extra)
        seed = int(j["name"].rsplit("_s", 1)[1])
        ns = M.parsed(j["args"])
        free = ("epochs",) if "A4" in used else ()
        # every non-attack setting is the winner's, except seed, tag, the shared pad and (A4) the cycle length
        assert M.non_attack(ns, free) == M.non_attack(wns, free), j["name"]
        assert (ns["seed"], ns["tag"], ns["vocab_pad"]) == (seed, j["name"], pad)
        if "A4" in used:
            assert "--epochs" not in j["args"]  # the cycle's length is its --sources plan
        # attack settings: the promoted arm's for the arms in this job, the trainer default (off) for the rest
        for k in M.attack_dests(ns):
            owner = M.attack_dest(k)
            want = arm_ns[owner][k] if owner in used else wns[k]
            assert ns[k] == want, (j["name"], k)
        # and token for token: the winner's args, in order, then the attack groups, seed, pad, tag
        rest = [g for g in M.groups(j["args"])
                if M.attack_of(g[0]) is None and g[0] not in ("--tag", "--seed", "--vocab_pad")]
        assert rest == [g for g in w_rest if not (free and g[0] == "--epochs")], j["name"]
        assert M.flat([g for g in M.groups(j["args"]) if M.attack_of(g[0])]) == \
            [t for k in M.ORDER if k in used for t in spec["arms"][k]["flags"]]


def test_loo_attributes_every_arm():
    spec = M.build({"A1": A1, "A2": A2, "A3": A3}, [0, 1, 2], loo=True)
    names = [j["name"] for j in spec["jobs"]]
    for label in ("a1a2a3", "a2a3", "a1a3", "a1a2", "a1", "a3", "base"):
        assert [f"stack_{label}_s{s}" for s in (0, 1, 2)] == [x for x in names if x.startswith(f"stack_{label}_s")]
    no_loo = M.build({"A1": A1, "A3": A3}, [0, 1, 2], loo=False)
    assert [j["name"] for j in no_loo["jobs"]] == ["stack_a1a3_s0", "stack_a1a3_s1", "stack_a1a3_s2"]


def test_refusals(tmp_path):
    with pytest.raises(SystemExit, match="at least two"):
        M.build({"A1": A1}, [0], loo=False)
    with pytest.raises(SystemExit, match="A3 \\+ A4"):
        M.build({"A3": A3, "A4": A4}, [0], loo=False)
    with pytest.raises(SystemExit, match="sets no A1 flags"):
        M.build({"A1": "babylm/configs/jobs_a1_r1_S.json:r1_sd_base", "A2": A2}, [0], loo=False)
    with pytest.raises(SystemExit, match="carries A2"):
        M.build({"A1": A2, "A3": A3}, [0], loo=False)
    job = M.load_job(A1)
    bad = {"name": "x", "module": job["module"], "args": job["args"] + ["--lr_scale", "1.0"]}
    f = tmp_path / "j.json"
    f.write_text(json.dumps({"jobs": [bad]}))
    with pytest.raises(SystemExit, match="outside attack flags"):
        M.build({"A1": f"{f}:x", "A3": A3}, [0], loo=False)


def test_main_writes_the_file(tmp_path):
    out = tmp_path / "s.json"
    M.main(["--arm", f"A1={A1}", "--arm", f"A2={A2}", "--seeds", "0", "--out", str(out)])
    d = json.loads(out.read_text())
    assert [j["name"] for j in d["jobs"]] == ["stack_a1a2_s0", "stack_a1_s0", "stack_base_s0"]
    assert all(j["module"] == "babylm.harness.train" and j["timeout_s"] >= 10800 for j in d["jobs"])


def test_own_g_gives_each_seed_its_own_generator():
    shared = M.build({"A1": A1, "A4": A4}, [0, 1, 2], loo=False)
    own = M.build({"A1": A1, "A4": A4}, [0, 1, 2], loo=False, own=True)
    assert "Shared-G screen" in shared["note"] and "Claim run" in own["note"]
    for j_sh, j_own in zip(shared["jobs"], own["jobs"]):
        seed = int(j_own["name"].rsplit("_s", 1)[1])
        a = dict(M.groups(j_own["args"]))
        assert a["--init_from"] == ([f"babylm/runs/a4_G_s{seed}/resume.pt"] if seed else ["babylm/runs/a4_G/resume.pt"])
        syn = [v for v in a["--sources"] if v.startswith("synthetic:")]
        assert syn == [f"synthetic:babylm/runs/a4_data{f'_s{seed}' if seed else ''}/syn_a4a:1"]
        # only the run directories change
        strip = lambda x: x.replace(f"_s{seed}/", "/") if seed else x
        assert [strip(t) for t in j_own["args"]] == j_sh["args"]
    assert "babylm/runs/a4_G_s1" in own["note"] and "babylm/runs/a4_data_s2" in own["note"]
    with pytest.raises(SystemExit, match="needs an A4 arm"):
        M.build({"A1": A1, "A2": A2}, [0], loo=False, own=True)


def test_singles_give_the_best_arm_comparator():
    # unpadded stack: the arms' own promotion runs already share its init stream, so no singles unless asked
    s = M.build({"A1": A1, "A3": A3}, [0, 1, 2], loo=False)
    assert [j["name"] for j in s["jobs"]] == ["stack_a1a3_s0", "stack_a1a3_s1", "stack_a1a3_s2"]
    s = M.build({"A1": A1, "A3": A3}, [0], loo=False, singles=True)
    assert [j["name"] for j in s["jobs"]] == ["stack_a1a3_s0", "stack_a1_s0", "stack_a3_s0"]
    # padded stack: the unpadded arms are re-run alone at the pad; A2's own runs are already padded
    s = M.build({"A1": A1, "A2": A2}, [0], loo=False)
    assert [j["name"] for j in s["jobs"]] == ["stack_a1a2_s0", "stack_a1_s0", "stack_base_s0"]
    assert all("16512" in j["args"] for j in s["jobs"])
    assert "best single promoted arm" in s["note"]
