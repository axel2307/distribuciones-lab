# Distribuciones Lab

An educational probability lab built with Python, Streamlit, SciPy, NumPy and Plotly. The interface and lessons are in Spanish, with Argentine wording. Choose a distribution, adjust its parameters, calculate an event, and connect the result to an illustrative worked example.

**[Open the public app](https://distribuciones-lab.streamlit.app/)** — deployed on Streamlit Community Cloud with Python 3.12.15, using `main` / `app.py`, without secrets. Public verification covered all 24 distribution screens and representative controls, including the edited binomial example and a very large event threshold. See the [verification record](docs/QA.md) for the precise scope.

[Guided question](docs/screenshots/guided-binom-question.png) · [Edited event and answer](docs/screenshots/guided-binom-event.png) · [Guided case on mobile](docs/screenshots/guided-t-mobile.png)

## What you can explore

- **8 discrete families:** Bernoulli, binomial, Poisson, geometric, negative binomial, hypergeometric, multinomial, discrete uniform.
- **16 continuous families:** normal, uniform, exponential, Gamma, Beta, lognormal, Student's t, chi-square, Fisher's F, Weibull minimum, Pareto type I, Cauchy, Laplace, Rayleigh, Gumbel maximum, GEV.
- Three tabs: **Explorar**, **Entender**, **Caso práctico** (the starting view). Every family retains its formula, symbols, support, assumptions, parameter effects, common mistakes and official SciPy reference.
- **24 guided cases:** a recognizable question and illustrative data with units, random variable and possible outcomes, contextual parameter meanings, 2–4 numerical calculation steps, a chart of the same event and a useful interpretation. Editing inputs updates the whole story in place, with an independent reset for each case.
- Student's t, chi-square and F start from sample sizes and measured summaries; degrees of freedom and the observed statistic are derived automatically. An advanced mode retains general parameter/event exploration without presenting arbitrary degrees of freedom as a sample design. Beta distinguishes uncertainty about a rate from individual purchases.
- The multinomial case explains the number of orderings and their probability, plots the complete count-vector event against its complement, and separates a single-channel marginal comparison.
- Interactive PMF/PDF and CDF charts tied to the calculated event. Controlled quantile windows show the probability outside the chart without renormalizing it.
- Lower and upper tails, inclusive intervals, discrete point probabilities and quantiles. Rare events use survival functions and logarithms.
- A dedicated multinomial view: editable categories, explicitly validated probabilities, expected counts, joint count-vector probability and a clearly labelled binomial marginal.
- Population moments distinguish finite, infinite, undefined and numerical-precision states. Models and cases retain their own state when you change distributions.

These 24 families are a teaching selection, not an exhaustive catalogue. Synthetic examples are explicitly illustrative; the app does not fit models to observations or produce forecasts.

## Install and run

Use **Python 3.12**. The direct dependencies below are pinned to versions installed and tested together; the application does not need network requests, credentials, paid services or an AI API during normal use.

```bash
git clone https://github.com/axel2307/distribuciones-lab.git
cd distribuciones-lab
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and start:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell activation is disabled, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

Tests cover all 24 defaults and worked cases, independent formulas, validation, monotonic CDFs, continuous quantile inversion, discrete quantile inequalities, inclusive/exclusive events, rare tails, degenerate distributions, heavy-tail moment thresholds, state isolation, multinomial constraints, plot data and real Streamlit AppTest interactions. GitHub Actions runs the suite on Python 3.12.

For restricted Windows sandboxes whose default temporary directory is not writable:

```powershell
New-Item -ItemType Directory -Path .tmp -Force | Out-Null
$env:TEMP=(Resolve-Path .tmp).Path
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp/pytest-run
```

Optional rendered browser verification (separate from AppTest):

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python -m streamlit run app.py --server.port 8505 --server.headless true
# In a second terminal with the same environment:
python tools/browser_smoke.py
python tools/case_browser_qa.py
```

Set `LAB_URL` to test another app URL. Set `LAB_BROWSER_EXECUTABLE` to use an installed Chrome executable instead of downloading Chromium. Browser tests create a fresh profile, not an authenticated personal profile; screenshots go in ignored `test-results/`.

The guided-case browser check defaults to `http://127.0.0.1:8506`; set `LAB_URL=http://127.0.0.1:8505` when using the server command above. It covers all 24 guided screens, an edited binomial event, refresh and mobile layouts. AppTest separately exercises edited inputs, reset, invalid-state recovery and sample-state persistence.

## Deploy to Streamlit Community Cloud

1. Push this directory as the root of a GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub. Review any requested permissions yourself.
3. Choose **Create app**, repository `axel2307/distribuciones-lab`, branch `main`, main file `app.py`.
4. In advanced settings select **Python 3.12**. No secrets are required. Deploy.
5. Wait for the build and test the deployed controls, formulas, charts, reset and example-loading actions. Starting a server is not sufficient verification.

See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). The requirements file is at the repository root; tests and browser automation are not runtime dependencies.

## Mathematical conventions

Full details: [MATHEMATICS.md](docs/MATHEMATICS.md), [official sources and mappings](docs/catalog-sources.md).

- Geometric counts **trials** until the first success, starting at 1. Negative binomial counts **failures** before `r` successes, starting at 0.
- Exponential uses a **rate** λ, mapped to `scale=1/λ`; Gamma uses shape α and **scale** θ.
- Lognormal parameters are the mean and standard deviation of `log(X)`, not the mean and standard deviation of X.
- Discrete uniform includes both endpoints; its upper bound maps to SciPy `high+1`.
- Pareto is **type I**, with threshold `xmin`; GEV uses the conventional ξ and maps SciPy `c=-ξ`.
- Variance means the strict central quantity around a **finite mean**. If no finite mean exists, variance and standard deviation are undefined. If a finite mean exists but the second moment diverges, they are infinite.
- Continuous density height is not a probability. An exact continuous point has probability zero. Discrete PMFs are plotted as bars, not interpolated densities.
- `Q(0)` and `Q(1)` use effective support endpoints; these may be infinite for a nondegenerate unbounded law. Degenerate masses return the same point for every q. Interior continuous quantiles that round to an endpoint are accompanied by an attainable-probability warning.

## Structure

```text
app.py                      Streamlit interface and callbacks
lab/catalog.py              Reusable registry, parameters, formulas and symbols
lab/calculations.py         Validation, SciPy mappings, probabilities and moments
lab/plots.py                Bounded Plotly views and event highlighting
lab/content.py              Shared educational foundations
lab/lessons_discrete.py      Discrete lessons and examples
lab/lessons_continuous.py    Continuous lessons and examples
lab/state.py                Independent model state
lab/case_narrative.py       Current worked-case interpretations
lab/case_studies.py         24 live contextual stories and units
lab/case_math.py            Numerical explanations from the active event
lab/case_inputs.py          Sample summaries → degrees of freedom/statistic
tests/                      Numerical, content, graphics and interaction tests
tools/browser_smoke.py      Optional actual-browser verification
tools/case_browser_qa.py    Guided-case desktop/mobile verification
docs/                       Mathematical conventions, sources and QA evidence
```

## Numerical and display limits

UI parameter bounds are practical exploration limits and are described beside each control; the mathematical domains are validated separately. At most 1,200 consecutive integer outcomes and 1,000 continuous sample points are plotted. Curves and shaded polygons approximate the display; probabilities come from the complete SciPy distribution. Singular density endpoints are omitted with an explicit notice, without replacing their density with a fictional finite value.

Floating-point arithmetic has limits. Positive probabilities that underflow are shown through their log-probability where available; if SciPy also loses the logarithm, the app explains the precision limitation. It does not silently present an underflow as an impossible event. Very heavy tails can require a practical viewing window, whose excluded probability is always reported. See [QA.md](docs/QA.md) for the verification scope and numerical and display limitations.

Very large discrete event thresholds are resolved exactly when the effective support determines the answer. Otherwise, unresolved integer cutoffs beyond `2**53-1` are rejected with a precision explanation; the app never silently substitutes a different event.
