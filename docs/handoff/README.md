# Session Handoff

## The rule

There is **one** handoff file: `docs/handoff/HANDOFF.md`. It is
**rewritten in full** at the end of every session — never appended,
never partially updated. The previous session's content is preserved
in git history (`git log docs/handoff/HANDOFF.md` shows every prior
version).

This rule is **unconditional**. Every session reads HANDOFF.md at
the start and rewrites it at the end. No exceptions for "small
sessions" or "interrupted work" — if work was done, the handoff
captures the resulting state; if work was interrupted, the handoff
captures where the interruption left things and what to resume.

## Why one file, rewritten

- **Predictable location.** Anyone (any PC, any tool) finds the
  latest state at exactly one path.
- **Forced concision.** Rewriting from scratch each session prevents
  accumulating stale notes that nobody reads.
- **Git is the archive.** `git log docs/handoff/HANDOFF.md` gives a
  full timeline; `git show <commit>:docs/handoff/HANDOFF.md` reads
  any past session's handoff.
- **One source of truth.** No "is the dated file or the current
  pointer right?" ambiguity.

## What every HANDOFF.md contains

1. **Session metadata** at the top — date, session number, short
   title.
2. **Summary** — one paragraph on what this session accomplished.
3. **State of the build** — branch, files added/modified,
   in-flight work (uncommitted changes, half-implemented features).
4. **Decisions made** — non-obvious choices with brief reasoning
   (the *why*, not the *what* — git diff shows *what*).
5. **Open questions** — anything next session needs to ask the user
   or investigate before proceeding.
6. **Next-up tasks** — concrete first steps for the next session,
   in priority order.
7. **Pointers** — sections of `paper/v3/spec.md`, specific commits,
   files to read first.
8. **Environment notes** — anything PC-specific or
   easily-overlooked-on-resume.

## Starting a new session

1. **Read `docs/handoff/HANDOFF.md`** first — before any other
   exploration.
2. Read the spec sections it points to (typically
   `paper/v3/spec.md`).
3. Check `git status` and `git log -5` for anything the handoff
   might have missed.
4. If anything is unclear, ask the user before proceeding.

## Ending a session

1. Commit any in-flight work, or document in HANDOFF.md why it's
   left uncommitted.
2. **Rewrite `docs/handoff/HANDOFF.md`** in full for the current
   session — increment the session number, update the date, replace
   the body.
3. Commit HANDOFF.md.
4. Push to remote so it's available on the user's other machines.

## Sessions number monotonically

Session N+1 always reads its starting context from session N's
HANDOFF.md (the version at HEAD when N+1 begins). If you need to
inspect older sessions:

```bash
# list all handoff revisions
git log --oneline docs/handoff/HANDOFF.md

# read a specific past session's handoff
git show <commit>:docs/handoff/HANDOFF.md
```

Session numbers never reset and never repeat.
