# 10-minute walkthrough outline (Round 2)

**1. Problem and principle (1 min).** Spot unjustified cost creep and explain it honestly. Code decides; retrieval
only suggests; "couldn't find one" beats a wrong reason.

**2. Core logic (2 min).** Monday weeks, ratio-of-sums cost per tonne-km. Own history = 8 prior weeks, no look-ahead,
fewer if unavailable. Peers = same type, same week, excluding itself. Proof: the 3 sample rows reproduce exactly.

**3. Flag rule and the 20% (2 min).** Rising AND (own >= 20% OR peer >= 20% with 26-week drift >= 10%). Why drift: the
trailing average hides the Mumbai-Pune creep. Why 20%: not in the brief; largest round value that still flags all
3 sample rows; show `logs/sensitivity.csv`.

**4. Notes and the gate (2 min).** 3 of 10 notes can justify a rise; 7 are traps. Show Chennai-Bangalore: 02-24 and
03-03 justified by N001, 03-10 and 03-17 unexplained (N001 expired, N009 says normal). Show N003 magnitude and the
8-week window, flagged as my assumption.

**5. Trust and cost (2 min).** 24/24 trap cases, 28 tests, justified-row audit, injection results (full recall from +25%,
0 wrongly justified), 3-run sha256 identical, 0 LLM calls / $0.

**6. Limits and next steps (1 min).** Baseline contamination (step: 2 of 4 weeks; creep: week 17), 2-route peer groups,
TF-IDF instead of embeddings. Next: clean baseline excluding flagged weeks, CUSUM, sentence embeddings.

## Questions to be ready for
- *Why not an LLM for the explanations?* Reproducibility and zero hallucination risk; template reasons are enough.
- *Why TF-IDF?* 10 short notes, offline and repeatable; the gate is what makes verdicts trustworthy.
- *What if N003 applied to 2025-05-12 on Mumbai-Delhi (+11.6%)?* The note says 5-7%; 11.6% exceeds 1.5x, so it stays unexplained.
- *What would you change with more time?* A clean baseline and a persistence rule.
