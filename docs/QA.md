# Verification record

Verified on **2026-10-14** on Windows with Python 3.12.10, NumPy 2.4.3, SciPy 1.17.1, Streamlit 1.55.0, Plotly 6.6.0 and Pytest 9.0.3. The source documentation was also checked against the current official SciPy and Streamlit references.

## Executed tests

```powershell
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp/pytest-release
.\.venv\Scripts\python.exe -m pip check
```

Result: **933 passed in 42.60s**; **No broken requirements found**. After the final visual adjustments, the graphics and interaction subset was repeated: **114 passed in 30.73s**.

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

## Publication boundary

The public GitHub publication uses the existing authenticated account `axel2307`. No new token or OAuth scope is required for that operation.

Community Cloud deployment must be completed and verified separately. The browser-control runtime for existing authenticated local sessions could not initialize (`MXC launcher: cannot locate required Windows platform directory`, HRESULT0x80070003), including after a reset/retry. A fresh test Chrome can inspect the Community Cloud sign-in page, but cannot supply the user's authenticated GitHub browser session. No password, new OAuth permission, payment or agreement was submitted. Until a successful build and the deployed app are verified, this project does **not** claim a live Community Cloud URL.

## Known limits

Calculations use double precision. A positive probability below its representable range is reported through a logarithm; if SciPy loses even that logarithm, the app reports insufficient numerical precision. Interior Beta quantiles can round to support endpoints; achievedCDFs are shown with a warning. Numerical shading approximates curves and cannot visually resolve all probability near an endpoint singularity. Plot windows and omitted probability remain explicit.
