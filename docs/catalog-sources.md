# Distribution sources and parameterisation evidence

Official SciPy reference pages consulted on **2026-10-14**. The online manual
identified itself as SciPy 1.18.0. This document records the conventions checked,
not a runtime dependency on the documentation website. The app works offline.
All continuous densities below are zero outside their stated supports; values
of a density at single endpoints do not change probabilities. Where a density
diverges at a boundary, the UI documents the omitted boundary point instead of
changing its probability law.

The symbol names below are public learning controls, not undisclosed SciPy
argument names. Practical sidebar bounds are separate from mathematical
restrictions. Default numerical moments are computed, not hardcoded prose.

| Family and official source | Convention verified and app mapping |
| --- | --- |
| [Bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html) | X is 0 or 1; success probability p ∈ [0,1]. `bernoulli(p)`; p = 0 or 1 is deterministic. |
| [Binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html) | X counts successes in n independent fixed-probability trials. `binom(n,p)`; n is a nonnegative integer and p ∈ [0,1]. Boundary probabilities use the continuous extension of the mass formula (0⁰ = 1). |
| [Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) | μ is the expected count in the specified interval, not a rate unless that interval is one time unit. `poisson(mu)`; μ = 0 gives mass one at zero. |
| [Geometric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html) | X counts **attempts including the first success**, with support 1,2,…; `geom(p)` and 0 < p ≤ 1. The number of failures would be X−1, a different variable. |
| [Negative binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html) | X counts **failures before r successes**, starting at zero. `nbinom(r,p)`; SciPy calls r `n`. The UI chooses integer r ≥ 1; SciPy's general positive real shape extension is acknowledged. Total attempts would be X+r. |
| [Hypergeometric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html) | M is total population, K marked elements, n draws without replacement. Map to `hypergeom(M,K,n)`; SciPy names its marked-elements argument `n` and its draw argument `N`. Support is max(0,n−M+K) through min(K,n). |
| [Multinomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html) | Joint vector of nonnegative integer counts summing to n, with probabilities summing to one. `multinomial(n,p)`. SciPy may replace the last probability if the sum is wrong: the app explicitly validates first and does not rely on this correction. Category i marginal is `binom(n,p_i)`. There is no scalar joint CDF in this app. |
| [Discrete uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html) | UI endpoints a and b are **both inclusive**. Map to `randint(a,b+1)` because SciPy's upper endpoint is exclusive. Probability of each integer is 1/(b−a+1). |
| [Normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) | `norm(loc=mu,scale=sigma)`; σ is standard deviation, not variance. These are population parameters; a sample mean or sample standard deviation estimates them. |
| [Continuous uniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html) | Endpoints a < b map to `uniform(loc=a,scale=b-a)`. SciPy scale is interval width, not upper endpoint. |
| [Exponential](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html) | UI λ is a positive **rate**, with `expon(loc=0,scale=1/rate)`. Expected wait is 1/λ. |
| [Gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html) | Shape α and **scale** θ map to `gamma(alpha,loc=0,scale=theta)`. A rate parameter would be 1/θ. Integer shape is the Erlang waiting-time interpretation. |
| [Beta](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html) | `beta(alpha,beta,loc=0,scale=1)` on [0,1]. Both shapes are strictly positive. A shape below one can cause an infinite endpoint density, not a point mass. |
| [Lognormal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html) | ln(X) is normal with mean μₗ and standard deviation σₗ. Map to `lognorm(s=sigma_log,loc=0,scale=exp(mu_log))`. Neither log parameter is a moment of X. |
| [Student's t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) | Central standard t: `t(df,loc=0,scale=1)`, df = ν > 0. No location or scale controls are exposed; this aligns with inference examples using a standardised statistic. Γ in the PDF is the gamma function. |
| [Chi-squared](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html) | Central chi-square: `chi2(df,loc=0,scale=1)` with ν > 0. It equals gamma with α = ν/2 and θ = 2. |
| [Fisher's F](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html) | `f(dfn,dfd,loc=0,scale=1)` with both degrees of freedom positive. It is (U/ν₁)/(V/ν₂) for independent chi-square variables; numerator and denominator degrees of freedom have distinct roles. |
| [Weibull minimum](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html) | Positive shape k and scale η: `weibull_min(shape,loc=0,scale=scale)`. SciPy calls k `c`. It is the minimum version; k = 1 is exponential. η is not the mean. |
| [Pareto](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html) | **Type I** with positive α and minimum xₘ: `pareto(alpha,loc=0,scale=xmin)`. The minimum is set by scale; shifting loc to xₘ would describe another law. |
| [Cauchy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html) | `cauchy(loc=x0,scale=gamma)`; x₀ is median and location, γ is positive scale. They are not mean and standard deviation, which do not exist. |
| [Laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) | `laplace(loc=mu,scale=b)`; b > 0. Its standard deviation is √2 b, while μ is both mean and median. |
| [Rayleigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html) | `rayleigh(loc=0,scale=sigma)`; σ > 0. In the norm-of-two-independent-normals interpretation, σ is each component's standard deviation, not the magnitude's. |
| [Gumbel right](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html) | **Maximum** version, `gumbel_r(loc=mu,scale=beta)` with β > 0. Standardised z = (x−μ)/β; μ is location (and mode), not mean. |
| [Generalised extreme value](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html) | App uses conventional ξ with F(x) = exp(−(1+ξz)^(−1/ξ)). SciPy uses the opposite shape sign, so map to `genextreme(c=-xi,loc=mu,scale=sigma)`. At ξ = 0 it is Gumbel right. Interior requires 1+ξz > 0; a finite boundary is included in the support closure. |

## Numerical interpretation

For univariate families, the CDF is F(x) = P(X ≤ x); the survival function is
P(X > x). A discrete inclusive interval uses F(floor(b)) − F(ceil(a)−1), with
the survival-function equivalent preferred in the upper tail. A continuous
interval uses F(b) − F(a), again switching to survival functions in the upper
tail. The app does not interpret a continuous density value as a probability.

For q ∈ (0,1), the discrete quantile is the smallest supported integer k with
F(k) ≥ q, while a continuous quantile inverts F. Endpoint quantiles and support
limits are handled separately to avoid exposing SciPy's discrete `ppf(0)`
lower-support-minus-one convention as a possible observation.

The technical links embedded in each lesson use the same official source URL
recorded here. These 24 families form a teaching selection rather than an
exhaustive catalogue of all probability distributions.
