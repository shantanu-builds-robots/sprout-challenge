# Rules

Fill in the bracketed parts before announcing.

## Prize

- **$20** to the entry with the best score on the competition arena (scoring in the [README](README.md#scoring)).
- Paid by [PayPal / other] within [14] days of the results. One prize; no runner-up prizes.

## Timeline

- **Opens:** [date]. The practice track and tools are public from then.
- **Deadline:** [date, time and time zone]. Late entries are not graded.
- **Results:** [date]. The arena and the grading seeds are published with the results.

## Entering

- Send [a link to a public GitHub repository / a zip file] containing your submission folder (`controller.py`
  plus any files it loads) to [address / form].
- One entry per person. A team counts as one person and shares the prize.
- You may update your entry until the deadline; the last version received counts.

## What's allowed

- Any control method: hand-written, optimisation-based, learned, or a mix.
- Your controller may use only what `reset` and `act` receive. Reading the simulator's state any other way, or
  interfering with the grader, the timing or other entries, disqualifies the entry.
- At grading time your code may use the Python standard library, numpy, scipy, onnxruntime and this package;
  bundle anything else you need inside your folder (pure Python only). There is no network access.
- Each `act` call should finish within 20 ms on one CPU core; late answers are discarded (see the README).
- Your entry must be your own work. Starting from the examples in this repository is fine.

## Judging

- The organisers grade every entry the same way, with `scripts/grade_all.sh` on the arena and secret seeds.
- If a bug in the competition code turns out to change results, the organisers may fix it and re-grade
  everyone.
- The organisers' decision is final. [Eligibility restrictions, e.g. age or country, if any.]
- By entering you agree that your entry and its score may be published.
