# Verification record

## Guided cases improvement — 2026-10-15

The prior case view had seven prose sections but no numerical derivation or event chart inside the case. Editing a parameter hid its context and replaced it with a generic description. The updated view opens on **Caso práctico** and connects five stages: a live contextual question, model/units, 2–4 substituted calculation steps, the same event on a chart, and an interpretation. Case inputs and explorer inputs remain independent and visibly identified.

Executed locally on Windows / Python 3.12.10 with the pinned dependencies:

- **1,236 tests passed in 121.77 s** (`python -m pytest -q --basetemp=.tmp/qa-tests/complete --tb=short --junitxml=test-results/qa-suite.xml`). The environment's `TEMP`/`TMP` were set inside the writable workspace. `pip check`: **No broken requirements found**.
- After stabilizing tab identity and mobile sidebar layout, the relevant AppTest/state subset was repeated: **113 passed, 56 deselected in 47.51 s**.
- Final review found close interval endpoints being merged by display rounding. The question, notation and numerical substitutions now preserve distinct input values. The affected journey/UI subset passed **162 tests in 36.73 s**; the final numerical/display subset passed **96 tests in 11.45 s**, including a positive Normal(50,2) interval from 50 to 50.00000001 and probabilities close to a parameter boundary. These are presentation regressions; probability evaluation remains in the unchanged engine.
- The original unmodified baseline was independently executed first: **1,048 passed in 67.74 s**.
- New checks cover all 24 authored cases and edited variants, contextual units, numerical substitutions, exact event highlighting, quantitative results, quantile jumps, underflow, reset, invalid-input recovery, state isolation, and sample persistence across distribution changes. The numerical engine and mathematical conventions are retained.
- Independent special-case values: t with n=16, mean 52 mL, s=4 mL and reference 50 mL gives SE=1 mL, t=2, df=15 and right-tail **0.03197250364236**; chi-square with n=10, s²=1.6 mm² and reference variance 1 mm² gives Q=14.4, df=9 and **0.10879095519401**; F from sample sizes 10/12 and variances 8/4 mm² gives F=2, df=(9,11) and **0.13856622748932**. Reversing the samples and using the opposite tail preserves that F probability.
- Multinomial: **2520 × 0.00003375 = 0.08505** for counts (2,3,5). Its primary chart represents that complete-vector event and its complement; a separate optional plot explicitly labels the binomial marginal. Beta probabilities refer to uncertainty about a conversion rate, not frequencies of purchases.

`tools/case_browser_qa.py` was run against the local app on `127.0.0.1:8506` using installed Chrome and a fresh temporary profile. It rendered all **24 guided cases** with no Streamlit exceptions or KaTeX errors and checked the edited binomial n=10, p=0.5, exact-five event (**0.24609375**), refresh, and desktop/mobile layouts. Screenshots and the JSON run record are in ignored `test-results/guided-cases/`. The existing connected-browser runtime could not initialize; this is a real browser test in a separate profile, not a claim of reviewing the user's authenticated browser session.

`tools/browser_smoke.py` also completed successfully: all **24 explorer screens**, probability and plotted-density updates, reset without leaving the current tab, every theory/case tab, the bounded heavy-tail F window, dark chart contrast, and a fresh 390 px mobile viewport. [Guided question](screenshots/guided-binom-question.png), [edited event](screenshots/guided-binom-event.png), and [mobile t case](screenshots/guided-t-mobile.png) are committed representative captures. These checks do not exhaust every parameter combination.

The complete suite logs and JUnit XML are kept locally in ignored `test-results/qa-suite.log` and `test-results/qa-suite.xml`.

## Guided cases publication verified — 2026-10-15

- Functional commit [`57f00f691c561378052a848dc86acdfe94daea6c`](https://github.com/axel2307/distribuciones-lab/commit/57f00f691c561378052a848dc86acdfe94daea6c) was pushed to `main`, and GitHub's commit API independently returned that exact branch head.
- [GitHub Actions run 38660579143](https://github.com/axel2307/distribuciones-lab/actions/runs/38660579143) finished successfully on Ubuntu / Python 3.12: **1,256 tests passed in 72.45 s**, including the final close-endpoint regressions. The dependency check also passed.
- The existing [public Streamlit app](https://distribuciones-lab.streamlit.app/) updated from the same `main` / `app.py` deployment. No repository, app, account permissions or access settings were changed; no new app was created.
- `tools/case_browser_qa.py` completed against that **public URL**, in a fresh Chrome profile, traversing all **24 guided cases**, the edited binomial (`n=10`, `p=0.5`, `P(X=5)=0.24609375`), refresh and mobile layouts. There were **zero Streamlit exceptions and zero KaTeX errors**. Public Community Cloud wraps the app in a frame; the verifier locates the rendered Streamlit surface rather than mistaking the outer page for the app.
- Public screenshots and the run's `results.json` are kept locally in ignored `test-results/public-guided-cases/`. Verification covers the tested scenarios, not all possible parameter combinations. The earlier deployment evidence below is historical.

## Previous release evidence — 2026-10-14

Verified on **2026-10-14** on Windows with Python 3.12.10, NumPy 2.4.3, SciPy 1.17.1, Streamlit 1.55.0, Plotly 6.6.0 and Pytest 9.0.3. The source documentation was also checked against the current official SciPy and Streamlit references.

## Executed tests

```powershell
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp/pytest-release
.\.venv\Scripts\python.exe -m pip check
```

Current result: **1048 passed in 64.71s**. `pip check` reported **No broken requirements found**. The original 934-test release passed in 43.02s; after its visual adjustments, the graphics and interaction subset was repeated: **114 passed in 30.73s**.

The follow-up fixes add 114 regressions: 47 current-case narrative tests (including independent altered scenarios for all 24 families and actual AppTest checks), 31 mathematical tests for enormous discrete event limits, and 36 AppTest checks for those limits and recovery. Edited cases now derive their interpretation from the current model and event, and mark the original context and justification as reference. Finite-support implications are resolved before SciPy; unresolved integer cutoffs beyond the safe precision range produce an explained error rather than a traceback.

GitHub Actions independently verified the initial 933-test release on Ubuntu with Python 3.12: **933 passed in 33.49s**, with a successful dependency check. [Executed CI run](https://github.com/axel2307/distribuciones-lab/actions/runs/38497284795).

The supervising task subsequently confirmed that [CI for functional commit `249a55c`](https://github.com/axel2307/distribuciones-lab/actions/runs/38499498227) completed successfully, covering the edited-case and large-event corrections.

Coverage includes all 24 defaults and complete worked examples; independent PMF/PDF calculations; the required Bernoulli/Binomial/die reference values; tiny SF/interval tails; monotone CDFs; quantile inversion/inequalities and endpoint precision; degeneracy repairs; strict central variance; mathematical domains and practical UI boundaries; multinomial vectors; chart bounds and exact event highlighting; isolated durable state, reset and actual Streamlit callbacks.

## Rendered browser checks

Used installed Chrome in a **fresh headless test profile**, without authenticated personal sessions. Streamlit served only on `127.0.0.1:8505`. These checks supplement AppTest, which does not verify client-side pixels.

- Selected each of the 24 families and visited its three tabs. All formulas rendered without `.katex-error`; all expected Plotly charts rendered without a Streamlit exception.
- Changed the normal mean from 0 to 1 with threshold 0: the displayed probability became 0.15865525, and the density peak moved to 1. Reset restored 0.5 and the original model.
- Inspected screenshots of density shading, stepped discreteCDFs, multinomial categories, GEV, and desktop light/dark themes.
- Detected and corrected low-contrast Plotly labels in dark mode. The verified chart text is RGB(250,250,250) against background RGB(14,17,23); text inherits the active theme through CSS, including the initial load.
- Checked a 390px mobile viewport. Long chart titles were shortened to fit.
- Heavy positive tails use a practical widening window; actual excluded probability is shown. SingularPDFs remain a visual approximation, explicitly explained, with exact numerical probability and a separately sampledCDF.

The reproducible browser script is `tools/browser_smoke.py`; screenshots are produced in ignored `test-results/`. Public example screenshots may be copied into `docs/screenshots/`.

## Verified public deployment

The [public GitHub repository](https://github.com/axel2307/distribuciones-lab) was published using the existing authenticated account `axel2307`. No new token or OAuth scope was required for that operation.

The supervising task confirmed the public app at **[distribuciones-lab.streamlit.app](https://distribuciones-lab.streamlit.app/)** on 2026-10-14. Community Cloud runs **Python 3.12.15**, with repository branch `main`, entrypoint `app.py`, and no secrets.

Verification was performed in the supervising task's cloud browser and covered all **24 public distribution screens** and representative controls. Functional commit `249a55c` was confirmed live with these specific regression checks:

- Edited binomial case: `n=10`, `p=0.5`, event `P(X=5)` gives **0.24609375**, with an interpretation describing the current 10 trials and exact-five event.
- Binomial event `P(X ≤ 1e20)`: the calculated probability is **1**, without a traceback.

This public verification supplements the local numerical, content, plotting and AppTest suites described above. It does not claim exhaustive testing of every parameter combination or control on the hosted app. The initial local authentication blocker was resolved by the supervising task; this documentation update did not initiate another deployment or change application behavior.

## Known limits

Calculations use double precision. A positive probability below its representable range is reported through a logarithm; if SciPy loses even that logarithm, the app reports insufficient numerical precision. Interior Beta quantiles can round to support endpoints; achievedCDFs are shown with a warning. Numerical shading approximates curves and cannot visually resolve all probability near an endpoint singularity. Plot windows and omitted probability remain explicit.
