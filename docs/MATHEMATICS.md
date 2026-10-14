# Mathematical and numerical conventions

The formula, symbols, controls, examples, SciPy mapping and plots describe one parametrization. Technical sources for each family are listed in [catalog-sources.md](catalog-sources.md).

## Sensitive mappings

| Family | Variable / parameters | SciPy mapping |
|---|---|---|
| Geometric | Trial number of first success, X≥1 | `geom(p)` |
| Negative binomial | Failures before r integer successes; total trials T=X+r | `nbinom(r,p)` |
| Hypergeometric | M population size, K population successes, n draws without replacement | `hypergeom(M,K,n)` |
| Discrete uniform | Integer bounds L≤X≤H, both included | `randint(L,H+1)` |
| Exponential | Rate λ>0, X≥0 | `expon(scale=1/λ)` |
| Gamma | Shape α>0, scale θ>0, X≥0 | `gamma(α,scale=θ)` |
| Lognormal | log(X)~Normal(μlog,σlog²) | `lognorm(σlog,scale=exp(μlog))` |
| Weibull minimum | Shape k>0, scale η>0, X≥0 | `weibull_min(k,scale=η)` |
| Pareto type I | Shape α>0, minimum xmin>0 | `pareto(α,scale=xmin)` |
| GEV | z=(x−μ)/σ; F(x)=exp(−(1+ξz)^(−1/ξ)) in the interior | `genextreme(c=-ξ,loc=μ,scale=σ)` |

GEV ξ=0 is the Gumbel maximum limit. ξ<0 gives a finite upper endpoint; ξ>0 gives a finite lower endpoint. An independent oracle checks ξ=−1, μ=2, σ=3 against X=5−Exp(scale=3): upper support 5, mean 2, variance 9, median 5−3 log(2).

## Events and tails

For integers in an inclusive real interval [a,b], L=ceil(a), U=floor(b). If L>U the event is empty; otherwise probability is F(U)−F(L−1). P(X>x)=SF(floor(x)). Discrete point queries require integer k, while tail and interval boundaries may be real. Highlighting uses the same comparisons.

Before calling SciPy, discrete events are compared to the effective support. This gives exact zero/one answers even for thresholds such as `1e20` when the law is bounded or a point mass. Interval endpoints may be reduced to the support only by removing impossible outcomes, which preserves the event probability. If a large cutoff remains unresolved on an unbounded support and exceeds the exact integer limit `2**53-1`, a clear precision error is raised; it is never clipped to an unrelated event or passed as an object-sized integer to NumPy/SciPy.

For continuous laws, P(a≤X≤b)=F(b)−F(a), and strict/inclusive point endpoints do not change its value. P(X=x)=0. In right-tail intervals the engine uses SF(a)−SF(b), with log differences when needed. No conceptual invalid input is repaired by numeric clipping; only tiny final floating-point roundoff outside [0,1] can be corrected.

Independent tail checks include Normal SF(10), Normal [9,10], Exponential rate1 [50,51], and Geometric p=0.5 [60,61]. Underflow is distinguished from an empty event using logarithms and support. If even a SciPy log function loses precision, a numerical-error message replaces the result.

## Quantiles and degenerate masses

For 0<q<1, discrete Q(q) is the smallest support integer satisfying F(Q−1)<q≤F(Q). Equality F(Q)=q is generally inappropriate. At q=0 and q=1, this app returns actual effective support endpoints, rather than SciPy's discrete lower−1 convention. Infinite endpoints are explained in the interface. Bernoulli p∈{0,1}, n=0, Poisson μ=0, Geometric p=1, Negative binomial p=1, fixed Hypergeometric and one-point integer Uniform cases are handled as point masses before trusting `ppf` or moment routines.

Continuous quantiles have finite-precision limits: for extremely singular Beta shapes, interior quantiles can round to 0 or 1. The interface reports the actual achieved CDF and warns when it differs from the requested probability.

## Moments

Variance is the strict central quantity E[(X−E[X])²], which requires a finite mean. This differs from references that call an extended second-moment divergence an infinite variance even without a finite mean.

| Family / condition | Mean | Central variance / standard deviation |
|---|---|---|
| Cauchy | Undefined | Undefined |
| t, ν≤1 | Undefined | Undefined |
| t, 1<ν≤2 | Finite (0) | Infinite |
| F, denominator df≤2 | +∞ | Undefined |
| F, 2<denominator df≤4 | Finite | Infinite |
| Pareto, α≤1 | +∞ | Undefined |
| Pareto, 1<α≤2 | Finite | Infinite |
| GEV, ξ≥1 | +∞ | Undefined |
| GEV, 0.5≤ξ<1 | Finite | Infinite |

Outside these cases, SciPy supplies numerical values, with explicit classification if a mathematically finite value cannot be represented. A median is computed as Q(0.5), without inventing a closed form; a discrete median may not be unique.

## Multinomial

There are 2–5 exclusive categories and a fixed integer n≥0. Inputs are validated before SciPy: every pᵢ∈[0,1], the sum equals1 within absolute tolerance1e−12 (only floating-point summation error), names are nonempty/distinct, integer kᵢ≥0 and sum(kᵢ)=n. No probability vector is normalized by the app. This matters because SciPy may replace its last category probability if inputs do not sum to1.

The joint PMF is n!·∏pᵢ^kᵢ / ∏kᵢ!. A zero-probability category with zero count contributes1. Expected counts are n·pᵢ. A selected marginal is Binomial(n,pᵢ). A category-probability chart is not the joint count distribution, and the app does not invent a scalar joint CDF.

## Charts

Display windows use support, quantiles and practical rendering limits, not a universal mean±3SD rule. Discrete windows keep consecutive outcomes, never subsample bars. The original law remains unchanged; omitted probability is CDF(left−1)+SF(right) for integer bars, or CDF(left)+SF(right) for continuous windows. Diverging PDF endpoints are omitted and described, while their area remains in the exact probability calculation. Numerical polylines and shading are a visual approximation, not the computation engine.
