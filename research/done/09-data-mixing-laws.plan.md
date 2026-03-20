# 09 — Data Mixing Laws for Code LLMs

## The Problem

When training a code LLM on multiple data sources (TypeScript, Python, JavaScript, etc.),
how do you decide the proportions? Should it be 60% TS + 30% Python + 10% JS? Or 80/10/10?
Getting this wrong wastes compute: too much of one language makes the model overfit to it,
too little of another means the model never learns it properly.

Historically, these proportions were set by vibes and gut feeling. Data Mixing Laws show
they can be predicted mathematically.

## Key Papers

### Data Mixing Laws (Ye et al., ICLR 2025)

**Core finding:** When you train a model on a mixture of K data domains, the final loss on
each domain can be predicted by a closed-form function of the mixing proportions. The
relationship follows a power law per domain:

```
L_i(p) = a_i * p_i^(-b_i) + c_i
```

Where:
- `L_i(p)` = loss on domain i given mixing proportions p
- `p_i` = proportion of domain i in the training mix
- `a_i, b_i, c_i` = domain-specific constants (fit from a few pilot runs)

**Why it matters:** Instead of running 100 experiments to find the best mix, you can:
1. Run ~10 pilot training runs with different proportions (small proxy model, short runs)
2. Fit the power law parameters (a, b, c) for each domain
3. Use optimization to find the proportions that minimize a target loss function
4. Apply those proportions to the full-scale training run

**Transferability:** The optimal proportions found on a small model (e.g., 50M params)
transfer to larger models. The absolute loss values change, but the relative ordering of
which mix is best stays remarkably consistent.

### DoReMi (Xie et al., NeurIPS 2023)

**Domain Reweighting with Minimax Optimization.** A two-stage approach:

1. **Stage 1 — Reference model:** Train a small model with uniform mixing (equal weights
   across all domains). This becomes the "baseline" that tells you where the model struggles.

2. **Stage 2 — Proxy reweighting:** Train another small model, but dynamically adjust the
   mixing weights during training. At each step:
   - Compute the loss of the proxy model on each domain
   - Compute the loss of the reference model on the same data
   - The "excess loss" = proxy_loss - reference_loss tells you which domains need more weight
   - Increase weights for high-excess-loss domains (minimax: help the worst-performing domain)

3. **Apply:** Use the final weights from Stage 2 for the full-scale training run.

DoReMi found that the optimal mix often differs dramatically from the natural distribution.
For example, in The Pile, the "Books" domain naturally makes up ~10% but DoReMi assigns
it ~3%, while "Wikipedia" goes from ~4% to ~12%.

### SlimPajama and RedPajama Mixing Strategies

**SlimPajama** (Cerebras, 2023): 627B token dataset derived from RedPajama with aggressive
deduplication. Their mixing proportions:
- CommonCrawl: 52.2%
- C4: 26.7%
- GitHub: 5.2%
- Books: 4.2%
- ArXiv: 4.6%
- Wikipedia: 3.8%
- StackExchange: 3.3%

Key insight: they kept GitHub code at only 5.2% despite having much more available, because
quality > quantity for code data.

**RedPajama-v2** took a different approach: provide quality signals per document and let
users decide their own filtering thresholds. Rather than prescribing a mix, give the
tools to find your own.

## Applying to Code Data

### Language Proportions

For a code-focused model, the "domains" are programming languages. Key considerations:

1. **Target language boost:** If the model is for TypeScript, TS should get the largest
   share. But not too much — cross-language transfer is real. Python code teaches general
   programming concepts that help TS generation.

2. **Language family transfer:** JavaScript and TypeScript are closely related, so JS data
   has disproportionate value for a TS model. Python and Ruby are less related but still
   teach algorithmic thinking.

3. **Data quality variance:** Not all languages have equal data quality. TypeScript code
   on GitHub tends to be more modern and well-typed than, say, PHP code from 2010. Weight
   should account for quality, not just quantity.

4. **Diminishing returns:** The mixing law power function shows diminishing returns.
   Going from 10% to 30% TypeScript has a much bigger impact than going from 60% to 80%.
   The sweet spot is usually in the 30-60% range for the primary language.

### Quality Tier Mixing

Beyond language, you can mix by quality tier:
- **Verified:** Code from repos with CI/CD, type-checking, tests passing
- **Tested:** Code from repos with test files present
- **Filtered:** Code that passes basic quality filters (length, diversity, syntax)
- **Raw:** Unfiltered code from bulk scrapes

A good starting mix: 30% verified, 25% tested, 30% filtered, 15% raw.
The raw data adds vocabulary diversity; the verified data teaches correctness.

### Source Diversity

- **GitHub:** Most volume, variable quality
- **StackOverflow:** Short, focused snippets with human-verified answers
- **Documentation:** High quality but repetitive
- **Textbooks:** Excellent for teaching concepts but limited volume

## Practical Approach for Cola-Coder

Given our scale (50M-350M params, single GPU), the full Data Mixing Laws pipeline
(fit power law curves from pilot runs) is overkill. Here is a more practical approach:

### Recommended Strategy: Inverse-Loss Reweighting

1. **Start with a preset** (e.g., `typescript_focused` = 50% TS, 25% JS, 15% Python, 10% other)
2. **Train for 1000 steps** with these proportions
3. **Measure per-source validation loss** — which language/source has the highest loss?
4. **Reweight:** Increase weight for high-loss sources, decrease for low-loss sources
5. **Repeat** once or twice (diminishing returns after 2-3 iterations)

This is a simplified DoReMi that skips the reference model and minimax optimization,
but captures the key insight: give more data to whatever the model struggles with most.

### Grid Search (Optional, More Compute)

If you have the patience for it:
1. Define 5-10 candidate mixing ratios
2. For each ratio, train a tiny model (50M) for 500 steps
3. Measure validation loss
4. Pick the ratio with the lowest average validation loss
5. Use that ratio for the real training run

This is more principled but costs 5-10x the compute of a single short run.

### What NOT to Do

- **Don't use natural proportions:** GitHub has way more Python than TypeScript. Using
  the natural distribution means your model barely sees TS.
- **Don't use equal weights for everything:** Some languages have much more noisy data
  than others. Equal weights = equal noise.
- **Don't optimize for a single language's loss:** You want the model to be good at TS
  but still competent at JS/Python. Optimize a weighted average of per-language losses.

## Integration with Cola-Coder

The `MixingConfig` and `MixingOptimizer` classes in `src/cola_coder/data/mixing.py`
implement the preset and inverse-loss approaches. They integrate with the existing
`DatasetCombiner` (combine.py) and `MixedSource` (sources/mixed.py):

```
MixingConfig (choose proportions)
  → DatasetCombiner.combine(weights=config.sources)  # for .npy files
  → MixedSource(sources, weights)                     # for streaming
```

The CLI integration adds a "Data Mixing Strategy" step to the interactive data
preparation menu, letting you pick presets or define custom weights.

## References

- Ye et al., "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling
  Performance" (ICLR 2025)
- Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"
  (NeurIPS 2023)
- Cerebras, "SlimPajama: A 627B Token Cleaned and Deduplicated Version of RedPajama"
  (2023)
- Together AI, "RedPajama-Data-v2: An Open Dataset with 30 Trillion Tokens" (2023)
