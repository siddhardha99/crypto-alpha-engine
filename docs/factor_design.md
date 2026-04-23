# Factor System — Phase 4 design

Answers to the five design questions before any `factor/*.py` code
lands. Committed alongside `ast.py` in the first Phase-4 commit so
the rationale travels with the code.

---

## 1. FactorNode shape

Already defined in Phase 1's `crypto_alpha_engine/types.py`; Phase 4
keeps that shape and adds a canonical-hash helper.

```python
@dataclass(frozen=True)
class FactorNode:
    operator: str                              # key into OperatorSpec registry
    args: tuple[ArgValue, ...]                 # positional args (see below)
    kwargs: dict[str, Any] = field(default_factory=dict)

ArgValue = FactorNode | str | int | float | bool
```

* **`args` representation.** Typed union `FactorNode | str | int | float | bool`, not a typed-ArgValue wrapper. Strings in `args` are *always* feature-name lookups at the AST level (our Phase 3 `arg_types` vocabulary declares which positions accept `"series"` — those are the positions where a string means "resolve via features_dict"). No operator in the current set takes a literal string param, so this unambiguous rule holds.

* **Vocabulary closure for the string-means-lookup rule.** The
  `arg_types` vocabulary is deliberately closed: `"series"`, `"int"`,
  `"float"`, `"bool"`, `"series_or_scalar"`. **There is NO `"string"`
  tag.** No operator may take a literal string as a positional arg.
  The "string in args → feature-name lookup" rule is load-bearing for
  the compiler's dispatch; documenting it as narrative would let it
  drift. We enforce it mechanically:

  * **In `ast.py`'s module docstring**: the vocabulary is enumerated
    with the explicit note that `"string"` is deliberately absent.
  * **In `tests/unit/test_ast.py`**: a canary iterates every
    registered operator and asserts no `arg_types` entry equals
    `"string"`. A future contributor who tries to add
    `@register_operator("foo", arg_types=("string",))` trips the
    canary immediately. (The registry's existing `VALID_ARG_TYPES`
    frozenset already refuses unknown tags at registration; this
    canary is the belt-and-suspenders check that the specific tag
    `"string"` never gets added.)
* **Identity and hashing.** Dataclass `frozen=True` gives us `__eq__` by value. `__hash__` is auto-generated but will raise at call time because `kwargs` is a dict. We don't fight this — we add an explicit `factor_id(node) -> str` helper in `ast.py` that serialises the tree to canonical JSON (sorted kwargs, stable float formatting) and returns `"f_" + sha256[:8]`. That's the factor identifier stored in the ledger and used as the cache key for compilation memoisation. Equality (`==`) is used for quick structural checks; `factor_id` is used wherever we need a stable string identifier.
* **Immutability.** Yes — `frozen=True`. `kwargs` is a dict, so deep mutation is possible; the style rule is "treat the node as immutable" and we don't rely on deep immutability except for hashing (which goes through `factor_id`).
* **`Factor` wrapper** stays as defined in `types.py`: name + description + hypothesis + root + metadata. Hypothesis is required so every stored factor carries the economic claim it tests, per SPEC §7.

---

## 2. Parser input format

**Primary authoring format:** Python-call-string, parsed via the stdlib
`ast` module. Example:

```python
ts_mean(funding_z("BTC/USDT:USDT", 24), 7)
```

**Serialisation format:** canonical JSON, used by the ledger and by
the AI research agents in Phase 2+.

```json
{"op": "ts_mean", "args": [{"op": "funding_z", "args": ["BTC/USDT:USDT", 24]}, 7]}
```

`parser.py` exports four functions:

* `parse_string(s: str) -> FactorNode` — authoring path
* `parse_json(s: str) -> FactorNode` — ledger-read path
* `serialise_to_json(node: FactorNode) -> str` — canonical, sorted keys
* `serialise_to_string(node: FactorNode) -> str` — round-trip the authoring form

**Why function-call strings for authoring:** reads like Python, familiar
to every contributor, and `ast.parse` + a visitor gives us a 30-line
parser instead of 300. S-expressions require re-training the author;
YAML is verbose and unergonomic for a deeply-nested factor; direct
`FactorNode(...)` construction is too noisy to use interactively.

**Validation at parse time.** The parser checks each `Call` node's
operator exists in the registry and the positional-arg count matches
the registered `arg_types` tuple. Unknown operators, arity mismatch,
or using a bare string where an `"int"` is expected — all raise at
parse time with line/column info from the Python `ast` node. This is
where the Phase 3 `arg_types` sweep starts paying off.

### Parser security — allowlist, not blocklist

Factor strings will eventually come from sources we don't fully
control: AI research agents in Phase 2+, ledger syncs from other
installations, community-contributed factor libraries. The parser
**never** calls `eval`, `exec`, or `compile` on any fragment of the
input. `ast.parse` builds a syntax tree only — no Python code is
executed at any point.

The visitor walks the tree node by node and rejects any node type
not in a strict allowlist:

* `ast.Module` — root wrapper from `ast.parse`.
* `ast.Expr` — the single expression-statement wrapper. Module body
  must contain exactly one of these (multi-statement input is
  rejected).
* `ast.Call` — function calls, the only shape a FactorNode takes.
* `ast.Name` — operator names, only when appearing as the `func` of
  a `Call`, in `ast.Load` context. The Name must be a registered
  operator; anything else (including dunder-named identifiers like
  `__import__`, `__builtins__`) fails at the "unknown operator"
  check before ever reaching the registry.
* `ast.Constant` — literal string / int / float / bool / None.
* `ast.Load` — required as the context on Name nodes.
* `ast.UnaryOp` with `ast.USub` or `ast.UAdd` **only**, and only
  when the operand is a numeric `ast.Constant`. This is the minimal
  exception that lets `ts_mean(x, -1)` parse (so it can reach the
  positive-window check that raises `ConfigError`). No other unary
  operator and no nested unary ops.

**Everything else is rejected** with a `ParseError` naming the
forbidden node type and its source location:

    ast.Attribute, ast.Subscript, ast.BinOp, ast.BoolOp, ast.Compare,
    ast.Lambda, ast.IfExp, ast.Dict, ast.List, ast.Set, ast.Tuple,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Starred, ast.Assign, ast.AugAssign, ast.Import,
    ast.ImportFrom, ast.keyword (kwargs deferred — see open question),
    every statement type, and any node type added to `ast` in a
    future Python version.

The allowlist is the security contract. New AST node types need
explicit code to be admitted; forgetting the allowlist entry means
the parser rejects a legitimate new feature — a safe failure mode.
A blocklist has the opposite failure mode (silent admission of
anything we forgot) and is therefore non-negotiably off the table.

**Parser module docstring** (text that lands verbatim in
`parser.py`'s module header):

> *This parser uses ast.parse to read the syntax tree only. It does
> not execute any Python code at any point. Factor strings are parsed
> structurally against an allowlist of AST node types.*

**Security test set** (test cases that must raise `ParseError`):

    "__import__('os').system('ls')"           # Attribute access
    "ts_mean.__class__.__bases__"             # Attribute chain
    "ts_mean([x for x in range(10)], 5)"      # ListComp
    "ts_mean(lambda x: x, 5)"                 # Lambda
    "ts_mean('x', 5); ts_mean('y', 5)"        # multi-statement
    "ts_mean('x', 5) + 1"                     # stray BinOp

Plus, as additional coverage:

    "ts_mean(__import__('os'), 5)"            # dunder Name
    "ts_mean('x', 5)[0]"                      # Subscript
    "ts_mean('x', 5); import os"              # Import

Each case gets a dedicated test with the specific forbidden node
type named in the `pytest.raises(..., match=...)` pattern, so a
future regression fires with a specific error message.

---

## 3. Compiler contract

Two-stage, not one-shot:

```python
def compile(factor: Factor) -> CompiledFactor:
    """Validate + build a closure. Side-effect free."""

class CompiledFactor:
    factor_id: str
    def __call__(self, features: dict[str, pd.Series]) -> pd.Series: ...
```

Rationale for two-stage: walk-forward backtests call the same factor
against many sliding data windows. Compilation (validation, cache
preparation) happens once; evaluation happens per window.

**`features` shape:** `dict[str, pd.Series]` with flat string keys. Convention: `"<symbol>|<column>"` — **pipe-separated, not colon-separated**. Examples:

    "BTC/USD|close"
    "ETH/USD|close"
    "BTC/USDT:USDT|funding_rate"      # ccxt perp symbol contains ":"
    "fear_greed|value"

Colon is reserved in the universe we live in — ccxt perp symbols like
`BTC/USDT:USDT` already embed one — so any colon-based separator is
ambiguous when splitting a feature key back into `(symbol, column)`.
Pipe (`|`) appears in no ccxt symbol and no column name we produce,
so `key.rsplit("|", 1)` is unambiguous by construction.

**Documented in `compiler.py`'s module docstring** so the engine
(Phase 6) and factor authors share one convention. The engine builds
the dict from loaded parquets; tests inject it directly. Single-
series datasets (F&G, DXY, etc.) use their natural key like
`"fear_greed|value"`.

**Arg-threading rule.** The compiler walks the AST tree and, for each `FactorNode`, resolves each positional arg by its Phase-3 `arg_type`:

| arg_type             | AST form                      | Resolution                                       |
| -------------------- | ----------------------------- | ------------------------------------------------ |
| `"series"`           | `FactorNode`                  | recursively evaluate                             |
| `"series"`           | `str`                         | `features[string]` lookup                        |
| `"int" / "float" / "bool"` | literal                 | pass through unchanged                           |
| `"series_or_scalar"` | `FactorNode` / `str` / number | recurse / lookup / pass through (dispatch on type) |

Positional order is preserved end-to-end. `kwargs` pass through literally.

**Caching / memoisation.** Yes, compile() builds a memoisation table keyed by `factor_id(subtree)` so a factor that references `"BTC/USD|close"` five times evaluates each sub-expression once. Cache scope is per-CompiledFactor (i.e., per compile call); there is no global cache across factors, which would get messy with the engine's forward-walking window mutations.

**Memoisation is a verified property, not a claim.** A dedicated test
in `tests/unit/test_compiler.py` wraps a kernel with a call counter,
compiles a factor that invokes the same sub-expression twice (e.g.
`add(ts_mean("x|close", 20), ts_mean("x|close", 20))`), and asserts
the wrapped kernel was called exactly once. If the memoisation
regresses silently, the test fires.

---

## 4. Similarity metric

**Hybrid: structural pre-filter + behavioural final check.**

* **Structural (ships in Phase 4):** normalised largest-common-subtree similarity on canonical-form ASTs. Define `sim(a, b) = |lcs_nodes(a, b)| / max(|a|, |b|)` where `|·|` is node count. Output in `[0, 1]`, symmetric, 1.0 for identical trees. Near-constant differences (same shape, different window parameter) score high — caught at the structural stage. Ships with SPEC §7's rejection threshold of 0.7.
* **Behavioural (hook in Phase 4, full impl deferred to Phase 6):** once a factor is compiled and evaluated on a validation window, we compute `|pearson_corr(factor_a_out, factor_b_out)| > 0.9` over the non-NaN intersection. Catches semantic duplicates (different tree shapes that produce the same output series). The Phase 4 module ships the function signature + tests against synthetic pairs; the hookup to the ledger / engine lands with Phase 7.

**Why hybrid.** Structural alone misses cases like `ts_mean(x, 20)` vs `ts_decay_linear(x, 39)` — different trees, closely correlated outputs. Behavioural alone is the correct semantics but expensive (needs data and compile + evaluate) and brittle around NaN windows. The structural pre-filter triages cheaply; behavioural resolves the ambiguous cases.

---

## 5. Complexity metric

Four ingredients from SPEC §7, each with a rejection threshold:

| Component        | Typical | Reject at |
| ---------------- | ------- | --------- |
| `ast_depth`      | 2–5     | > 7       |
| `node_count`     | 3–15    | > 30      |
| `unique_operators` | 2–5   | (soft)    |
| `unique_features`| 1–3     | > 6       |

**Scalar complexity score** (used by the Deflated-Sharpe adjustment in Phase 5):

```
complexity = 0.4 * (depth / 7)
           + 0.4 * (node_count / 30)
           + 0.2 * (unique_features / 6)
```

Depth and node-count weighted equally at 0.4 each; features at 0.2 (fewer distinct features → lower overfit risk, but the main driver is tree shape). Output is in `[0, 1]` with 1.0 right at the rejection boundary.

**Typical values** (compute for a small menagerie during implementation — these are the *expected* order-of-magnitudes):

* `ts_mean("BTC/USD:close", 20)` — depth 2, 3 nodes, 1 feature → **~0.18**
* `ts_zscore(ts_diff("BTC/USD:close", 1), 20)` — depth 3, 5 nodes → **~0.24**
* Deeply nested 5-level, 20 nodes, 4 features → **~0.69**
* Just above reject threshold (depth 8, 31 nodes, 7 features) → **>1.0** (rejected before scoring)

So `complexity` for accepted factors is empirically in `[0.15, 0.85]`. The Deflated Sharpe adjustment in Phase 5 can treat this as a scalar penalty multiplier.

**`complexity.py` exports**:

```python
def ast_depth(node: FactorNode) -> int
def node_count(node: FactorNode) -> int
def unique_operators(node: FactorNode) -> set[str]
def unique_features(node: FactorNode) -> set[str]   # string args at "series" positions
def factor_complexity(node: FactorNode) -> dict[str, Any]   # all four + the scalar
def reject_if_too_complex(node: FactorNode) -> None   # raises ConfigError on violation
```

---

## What Phase 5+ inherits

* The `factor_id` string is the ledger key (Phase 7).
* The `features_dict` shape + key convention is what the Phase 6 engine populates before calling `CompiledFactor`.
* The behavioural-similarity hook needs a reference validation window — Phase 5 provides it via `DataSplits.validation_end`.
* The scalar complexity output feeds the Deflated Sharpe calculation in Phase 5's `statistics/deflated_sharpe.py`.

## Positional-only — confirmed

Phase 4 parser does not accept keyword arguments. `ast.keyword` is
**not** in the security allowlist. Operators whose positional order
isn't obvious (`ts_quantile(x, window, q)`, `clip(x, lo, hi)`) rely
on the registered `arg_types` tuple and the operator's own docstring
for documentation. If a contributor proposes kwargs later, it becomes
a deliberate design change requiring (a) a `kwarg_types` field on the
registry, (b) a new allowlist entry for `ast.keyword` under strict
rules, and (c) updates across parser, compiler, and similarity — not
a drive-by addition.
