# Notes — the channel back from whoever ran the job

The lab records what it did. This records what its user made of it.

That gap is not theoretical. A review on 2026-08-26 read seven days of the event ledger next to
the consuming project's own campaign logs, and found that its three most expensive discoveries had
been written down carefully — in *its* repo, as prose — and none had ever reached the lab:

- a `--price-cap 0.85` that billed $1.39–2.22/hr, booking **$11.88 against a $10.00 hard cap**,
  which cost a planned experiment that was never run;
- a catalog error that blamed **price** when the real cause was the accelerator **name**
  (`RTX_4090` vs `RTX4090:1`), costing two submits to diagnose;
- `sweep-aggregate` crashing on duplicate seeds, so five published points were assembled by hand.

The ledger could not hold any of them. It knows what was called and what came back — never what a
person concluded. Notes are that missing half.

## Writing one

```bash
# about a job
uv run lab note 20260825-163005-b037e0 --kind "BUDGET EVENT" --usd 11.88 \
    -m "three shards billed \$1.39-2.22/hr against --price-cap 0.85"

# about a submit that never became a job — often the best notes
uv run lab note --kind GOTCHA \
    -m "the catalog error blamed price; the real cause was RTX_4090 vs RTX4090:1"

# attach it to the failure you just hit, so the next run that hits it sees this
uv run lab note --last -m "job ids are full timestamps; do not truncate the paste"
```

`--kind` is free text. These are the ones already in use, so a reader can group on them:
`GOTCHA`, `BUDGET EVENT`, `ROOT CAUSE`, `INCIDENT`, `LESSON`, `DEVIATION`, `FEATURE REQUEST`,
`NOTE`. `--usd` records what it cost, which is what makes one note rankable against another.
`--agent` marks an agent as the author, so a reader can weight it.

**Write one when** a cost or duration differed from what you were told to expect; an error message
pointed at the wrong cause; you worked around the lab rather than with it (a hand-rolled watchdog,
a polling loop, manual aggregation — say what pushed you there); something in the skill turned out
to be stale; or a result is untrustworthy for a reason the manifest cannot show.

## Why `--last` exists

A note is found again by its **signature** — `lab.events.stats.signature`, the same normalisation
`lab report` groups by, which strips job ids, zones, paths and magnitudes so the key survives into
the next occurrence.

You cannot type that signature. The ledger masks an error message *before* signing it, so the
signature of what your terminal printed is not the signature the digest computes:

```
printed : unknown job id '20260823-0936'
ledger  : unknown job id 20260823-0936
signature: Exit: unknown job id <sha>-<n>
```

`--last` reads it back off the ledger. Without it the field would only ever be filled by someone
who had read `events/stats.py` and guessed right — which is to say, never.

## Reading them back

```bash
uv run lab notes                  # this project
uv run lab notes --all-projects   # everywhere this machine has run jobs
uv run lab notes <job_id>         # one job's
uv run lab notes --format md      # a TEAM-LOG-shaped table to paste
```

Notes also appear as a `notes` count on `lab status`, and as `notes.jsonl` inside
`lab export` bundles. That last one matters: `runs/` is git-ignored, so the bundle is the only
route a note has into the repo where the result is written up — which is exactly where
*"two near-threshold cells flipped across attempts"* needs to be legible.

## The push

When a call fails and a note matches its signature, the note is printed to **stderr**:

```
$ uv run lab status 20260825-1630
{
  "error": "unknown job id '20260825-1630'"
}
[lab] a previous run left a note on this:
  · GOTCHA (2026-08-26, human) job ids are full timestamps; do not truncate the paste
  (`lab notes --retire <id>` when one of these stops being true)
```

Note that the note was filed against a *different* truncated id. The signature is what makes it
carry over.

Stdout keeps only JSON, and the exit code is unchanged. The push is silent unless a **signed**
error matches: an errorless failure signs as the literal `"unknown"`, which would match every
unsigned note, so that case is refused outright rather than allowed to become a nag.

## Retiring — the part that keeps this honest

```bash
uv run lab notes --retire n-1a03ebbbb47-8472 --reason "enforced on-box since v0.1.0"
```

Do this the moment a note stops being true. A channel that never retires anything is a machine for
distributing folklore *at scale* — which is precisely the failure it was built to stop. The
consuming project still runs a hand-written `vastai` watchdog against a wall-clock cap that has
been enforced on the instance since v0.1.0, and still guards against a `reconcile` that has been
unable to touch another project's resources since v0.7.0. Automating the spread of that would be
worse than having no channel.

Two things make staleness visible even with nobody curating:

- every note records the `lab_version` it was written at;
- the push dates a note from a different version inline — *"lab 0.1.0 — you are on 0.9.0"*.

Treat that clause as a reason to check a note, not to obey it.

Retired notes stay readable (`lab notes --include-retired`) because they are history. They are
never pushed at anyone again.

## Where the files are, and who can read them

| | |
|---|---|
| Per job | `runs/<job_id>/notes.jsonl`, beside `logs.txt`; travels into `lab export` |
| Global index | `~/.lab/notes/index.jsonl` (`LAB_NOTES_DIR` overrides) |
| Disable | `LAB_NOTES=0` |
| Debug | `LAB_NOTES_DEBUG=1` surfaces a swallowed write error |

The index is user-global on purpose. The consuming project runs `lab` from one checkout
(`tempotron-capacity`) and does its thinking in another (`snn-research`); a project-local store
would be invisible from the repo where the conclusion gets written. Every note is project-tagged,
so per-project filtering is a read-side concern.

**Nothing is uploaded anywhere.** These are local files. Note text passes through the same
secret masking as the event ledger (FR-J1), but do not paste credentials into one on purpose.

Best-effort throughout, like the ledger: a note that cannot be filed will never fail the command
that was trying to file it.

## The loop this closes

```
someone hits a surprise
   -> lab note --last
      -> the next run that hits it gets the note on stderr
         -> at release: durable ones are promoted into the skill's
            "Corrections" section, and the note is retired
            -> lab init warns that the skill changed, and says to re-read it
```

The skill outranks a note. A note is *evidence from a previous run*, not doctrine — and a note
that contradicts the skill is itself worth a note.
