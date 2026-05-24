# trioron — Claude project instructions

## The handoff rule (unconditional)

This project uses a single handoff file at
`docs/handoff/HANDOFF.md`, **rewritten in full at the end of every
session**. The user works across multiple PCs; memory files in
`~/.claude/projects/.../memory/` do not sync. The handoff file is
the cross-PC source of truth.

**This rule has no exceptions.** Every session MUST read
HANDOFF.md at start and MUST rewrite it at end. Even for short
sessions, even for sessions that only investigate without changing
code — the handoff is rewritten so the next session knows what was
done and what is current. Skipping the handoff once breaks the
cross-PC chain.

**At the start of every session:**

1. Read `docs/handoff/HANDOFF.md` — *before* exploring the
   codebase, *before* answering questions, *before* anything else.
2. Read the spec sections it points to (typically in
   `paper/v3/spec.md`).
3. Check `git status` and `git log -5` for anything the handoff
   might have missed.
4. If anything is unclear, ask the user before proceeding — don't
   guess across a session boundary.

**At the end of every session:**

1. Commit any in-flight work, or document in the handoff why it is
   left uncommitted.
2. **Rewrite** `docs/handoff/HANDOFF.md` in full — increment the
   session number, update the date, replace the body. Do not
   append; do not edit selectively.
3. Commit the handoff.
4. Push to remote.

The previous session's handoff is preserved in git history; no
dated archive files are kept. See `docs/handoff/README.md` for the
content schema.

## The spec is the source of truth

`paper/v3/spec.md` is the authoritative Trioron v2.0 architecture
spec (9 sections, ~3100 lines). Any v2 question should be answerable
from the spec; if it isn't, that's a spec gap to discuss with the
user before writing code.

Section 9.14 has a cross-section index for fast lookup. Section 9
has the directory partition. Section 6 has the binding performance
contract (Phase 1 = 50K params, CPU, full-fidelity recording).

## Avoid v1's patch sprawl

The v1 codebase's pain (node.py = 2551 lines, api.py = 1446,
dreaming.py = 1415, phase-2.5/5/6 patch files) is what motivated
v2.0. Maintain the v2 discipline:

- One concept per file
- If a file is growing past ~500 lines, split along a concept boundary
- New patches files like `*_phase_*.py` are forbidden by convention
- The partition map (spec §9) is updated *before* code is added that
  would change it

## v1 modules

The v1 code lives in `trioron/legacy/` (moved via `git mv`,
unchanged). The compat layer (`trioron/compat/`) bridges v1 callers
to v2 implementations. v1 tests under `tests/test_*.py` continue to
pass and are part of the CI gate.

## Phase 1 binding constraints

- Maximum 50K parameters per substrate during Phase 1 (enforced at
  the API surface)
- CPU only — GPU is optional and non-contractual
- Full-fidelity recording (snapshot every growth event)
- CI hard-ceiling on bench targets = release blocker
- Phase 2 (1M params, sampled recording) is stretch / non-binding

## User preferences

The user (Marcelinus, addressed as "Chloe" in responses per global
CLAUDE.md) prefers:

- Brief, direct communication
- Sometimes-vague requests — ask clarifying questions rather than
  guessing
- Iterative review for non-trivial work (write 1–2 spec sections /
  one code module, summarize + flag decisions, wait for sign-off)
- Plain corrections — "If I'm wrong, say so plainly"
- Exact commands shown over descriptions
- No filler / no validation padding
