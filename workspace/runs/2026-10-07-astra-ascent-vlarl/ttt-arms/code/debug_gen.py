"""Print raw episode transcripts to sanity-check task difficulty and verification."""
import os, sys
import taskgen as tg
import harness as h

def main():
    mdl = h.Model.load()
    cfg = h.Config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    fam = sys.argv[3] if len(sys.argv) > 3 else None
    stream = [t for t in tg.build_stream(seed=seed, n_tasks=40) if fam is None or t.family == fam]
    shown = 0
    for t in stream:
        if shown >= n: break
        st = t.new_state()
        msgs = [("user", t.prompt())]
        ctx = mdl.render(tg.SYSTEM, msgs)
        print("="*90)
        print("TASK", t.tid, "lvl", t.level, "| expected:", {k: v for k, v in t.spec.items() if k in ("answer","conv","unit","clauses","chain")})
        print("PROMPT>>>", t.prompt()[:500].replace("\n"," | "))
        for turn in range(t.max_turns):
            text, ids, fl, ents = mdl.generate(ctx, cfg.max_new_tokens)
            kind, payload = tg.parse_turn(text)
            valid, env, fin, ok = tg.step_env(t, st, kind, payload)
            print(f"  turn{turn}: [{kind}] valid={valid} text={text!r}")
            if valid is False: print(f"    env: {env}")
            elif env: print(f"    env: {env}")
            if fin:
                print(f"  -> FINISHED success={ok} ({env})")
                break
            msgs.append(("assistant", text)); msgs.append(("user", f"Environment: {env}"))
            ctx = mdl.render(tg.SYSTEM, msgs)
        shown += 1

main()
