# Which SemBench queries compile to a Code Generator result?

A semantic operator is **compilable** when its predicate depends on a *nameable,
extractable slot* on the unstructured side, and the other side is already
structured (or extractable into the same slot). If so, the operator becomes:
extract the slot once → run relational code → residual model call only on the
unsure rows. If not (open-ended generation with no stable schema, or a predicate
that is irreducibly joint), it stays naive.

This is the exact test the **Schema Designer** applies (`agents/schema-designer/prompt.md`).

## MMQA queries (files/mmqa/query/bigquery)

| Query | Operator | Modality | Compilable? | Extracted schema slot | Notes |
|-------|----------|----------|-------------|-----------------------|-------|
| **q2a** | `AI.IF` join (racetrack × image logo) | image + text | **Yes (high value)** | `images.logo_brand` | Asymmetric — extract image side only, hash-join. M×N → N. |
| **q7**  | `AI.IF` join (airline × image logo) | image + text | **Yes (high value)** | `images.logo_brand` | Same pattern as q2a; **reuses the same `img_attrs` table** — zero extra extraction. |
| **q2b** | join then `AI.GENERATE` logo color | image + text | **Yes** | `images.logo_brand`, `images.logo_color` | Add one attribute to the same schema; both the join and the map read `img_attrs`. |
| **q1**  | `AI.GENERATE` director from text, then join on Title | text | **Yes** | `movie.director` | The map *is* the extraction; downstream is a plain equi-join. |
| **q3a–g** | `AI.IF` genre filter (comedy/sci-fi/horror/…) | text | **Yes (high value)** | `movie.genres[]` | **7 queries share one extraction.** Each becomes `'<genre>' IN genres`. Best amortization case in the set. |
| **q4**  | `AI.GENERATE output_schema=ARRAY<STRING>`, UNNEST + GROUP BY | text | **Yes** | `movie.genres[]` | Already schema-shaped; the map result feeds ordinary aggregation. Same slot as q3. |
| **q5**  | `AI.IF` "has flights to Frankfurt" over `Destinations` | text (structured) | **Yes (marginal)** | `airline.serves_regions[]` | Data is already semi-structured; extract a region set once, then set-membership filter. |
| **q6a–c** | `AI.IF` "has flights to Europe/…" | text (structured) | **Yes (marginal)** | `airline.serves_regions[]` | Same slot as q5; siblings amortize the one extraction. |

### Reading of the table
- **Flagship (multimodal join):** q2a / q7 / q2b — the M×N vision cost is the whole
  ballgame, and it collapses to one extraction pass over the images plus a hash
  join. These are the queries the user's write-up targets, and the PoC implements
  exactly this.
- **Best amortization:** q3a–g (+q4) — a single `genres[]` extraction serves 8
  queries. Extraction cost is paid once and divided across the whole family.
- **Marginal:** q5 / q6 — compilable but the input is already text-structured, so
  the win is smaller (you trade a per-row LLM filter for a per-row extraction; the
  gain is amortization across the q6 siblings, not a M×N collapse).

## Generalizing to the other scenarios

| Scenario | Compilable pattern | Extracted schema | Value |
|----------|--------------------|------------------|-------|
| **E-Commerce** (14 q: filter/join/map/rank/classify, image+text) | product image → attributes; join/filter/rank on them | `product_attrs{category, color, material, condition, defect}` | High — 14 queries over one product corpus. |
| **Wildlife** (10 q: filters, image+audio) | image/audio → species/behavior tags | `obs_attrs{species, behavior, count}` | High — all filters read the same tags. |
| **Cars** (10 q: filters+classify, image+text+audio) | damage image → structured damage report | `damage_attrs{part, severity, type}` | High — classify/filter compile to predicates on the report. |
| **Movie** (10 q: filter/join/rank/classify, text) | review/description → sentiment, genre, entities | `movie_attrs{sentiment, genres[], director}` | Medium — text only; still amortizes across the family. |

## The general rule
Compile when the operator's predicate factors through a **per-item attribute** the
corpus can be extracted into **once**. The multimodal join (logo, product match)
is the highest-value case because the naive plan is quadratic (M×N model calls);
compilation makes it linear in the corpus (N extractions) plus a hash join.
Filters/classify over a shared corpus win through amortization (one extraction,
many queries). Rank/map lower the same way. Only irreducibly-joint or open-ended
predicates with no stable slot stay naive — and SemBench has essentially none of
those.
