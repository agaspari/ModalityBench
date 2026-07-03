# FCT — Flattened Contextual Tree

> **Status:** implemented (Phase 2). Spec below is the user's original definition, saved
> verbatim as the source of truth. The implementation in
> `modalitybench/observations/serializers/fct.py` must reproduce this format exactly,
> including the header row and ditto (`"`) scope compression.

## Definition

**Tree semantics without tree syntax.** Every node is a pipe-delimited record; hierarchy
survives as a **scope column** with **ditto compression** (`"` = same scope as the previous
row).

This is the most tokenizer-predictable format — every row costs a near-constant token count,
which matters when doing budget math per page. It is also the best format when the model must
do set operations ("check all boxes in scope=filters"), because rows are trivially filterable.

## Columns

```
ref|t|name|val|flags|scope
```

- `ref` — opaque element id used by actions.
- `t` — two-letter type code: `tx` text, `ln` link, `in` input, `se` select, `ck` checkbox,
  `bt` button, `rd` radio, `im` image, etc.
- `name` — accessible name / visible label.
- `val` — current value (empty when none).
- `flags` — comma-separated: `req`, `empty`, `off`, `primary`, `dis` (disabled), `max4`,
  `opts=243`, `done`, …
- `scope` — nearest section label; `"` means "same scope as the row above".

## Example — checkout payment step

```
ref|t|name|val|flags|scope
e1|tx|3 items · $142.50|||sum
e2|ln|Edit shipping||done|ship
e3|in|Card number||req,empty|pay
e4|in|Expiry MM/YY||req,empty|"
e5|in|CVC||req,empty,max4|"
e6|se|Country|United States|opts=243|"
e7|ck|Save card|off||"
e8|in|Promo code||empty|"
e9|bt|Apply|||"
e10|bt|Pay $142.50||primary,dis|act
e11|ln|Return to cart|/cart||act
```

## Implementation notes

- Always emit the header row `ref|t|name|val|flags|scope` first.
- Pipe (`|`) characters inside any field are escaped (e.g. replaced with `∣` or `\|`) so the
  record shape stays parseable.
- The scope column emits the literal section label on the first row of a scope, then `"` for
  each subsequent row sharing that scope — reset when the scope changes.
