# Rules

The Sprout Challenge celebrates 1,000 followers on Instagram ([@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/)).
It's a community challenge: build a controller, race it, and compare notes with everyone else on the page. The best
runs get shared on the account, and the fastest one wins a small thank-you prize. It is a contest of skill, with
no luck draw and nothing to buy.

## Prize

- **A $20 gift card** (Amazon, or an equivalent gift card available in the winner's country) for the entry with
  the best score on the competition arena (scoring in the [README](README.md#scoring)).
- Sent within 14 days of the results to the email address the winner gives by DM. One prize; no runner-up prizes.
  The prize can't be exchanged for cash.

## Timeline

- **Opens:** the day the announcement reel is posted on Instagram. The practice track and tools are public from then.
- **Deadline:** **30 November 2026, 23:59 IST (UTC+5:30).** Late entries are not graded.
- **Results:** by 3 December 2026, on Instagram. The arena and the grading seeds are published with the results.

## Entering

- Follow [@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/) on Instagram, and still follow
  it at the deadline.
- DM [@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/) a link to a **public GitHub
  repository** containing your submission folder (`controller.py` plus any files it loads).
- One entry per person. A team counts as one person and shares the prize.
- You may update your entry until the deadline; the version in the repository at the deadline counts.
- Anyone, anywhere, of any age may enter, unless the law where you live forbids it. If you are under 18, a parent
  or guardian must agree to your entry and receives the prize on your behalf.

## What's allowed

- Any control method: hand-written, optimisation-based, learned, or a mix.
- Your controller may use only what `reset` and `act` receive. Reading the simulator's state any other way, or
  interfering with the grader, the timing or other entries, disqualifies the entry.
- At grading time your code may use the Python standard library, numpy, scipy, onnxruntime and this package;
  bundle anything else you need inside your folder (pure Python only). There is no network access.
- Each `act` call should finish within 20 ms on one CPU core; late answers are discarded (see the README).
  No GPU is needed, for training or for grading.
- Your entry must be your own work. Starting from the examples in this repository is fine.

## Judging

- The organiser grades every entry the same way, with `scripts/grade_all.sh` on the arena and secret seeds.
- If a bug in the competition code turns out to change results, the organiser may fix it and re-grade everyone.
- The organiser's decision is final.
- By entering you agree that your entry, your Instagram handle and your score may be published.

## Instagram

This competition is not sponsored, endorsed or administered by, or associated with, Instagram or Meta. By entering,
you release Instagram and Meta from any responsibility for it. Questions go to
[@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/), not to Instagram.
