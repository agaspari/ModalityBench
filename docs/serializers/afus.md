# AFUS — Abstract Functional UI Syntax

> **Status:** implemented (Phase 2). Spec below is the user's original definition, saved
> verbatim as the source of truth for the serializer. The implementation in
> `modalitybench/observations/serializers/afus.py` must reproduce this grammar and be
> validated against the checkout example (~200 tokens).

## Definition

An **affordance-oriented line grammar**. Every line is either a scope marker or an actionable
element:

```
TYPE:REF "AccessibleName" [flags]
```

All sigils are short lowercase ASCII words that tokenize to 1–2 tokens; **no indentation**
(whitespace costs tokens and models miscount it anyway — the 2D point applies to indentation
too).

## Example — checkout payment step (~200 tokens)

```
@page "Checkout · Payment" /checkout/payment step=2/3
@sec summary
txt "3 items · total $142.50" more:e1
@sec shipping done
txt "J. Doe · 123 Main St · Chicago IL"
lnk:e2 "Edit"
@sec payment active
in:e3 "Card number" req empty numeric
in:e4 "Expiry MM/YY" req empty
in:e5 "Security code" req empty max=4
sel:e6 "Country" ="United States" opts=243
chk:e7 "Save card" off
in:e8 "Promo code" empty
btn:e9 "Apply"
@sec actions
btn:e10 "Pay $142.50" primary disabled
lnk:e11 "Return to cart" /cart
~nav "Footer links" x=14 expand:e12
```

## Grammar notes (derived for the implementation)

- **Scope markers** start with `@` (e.g. `@page`, `@sec`) or `~` (collapsed/low-priority
  region, e.g. `~nav`). A scope marker may carry a name string and trailing flags.
- **Element lines** are `type:ref "name" [flags...]`. `type` is a short affordance sigil:
  `btn` (button), `lnk` (link), `in` (text input), `sel` (select), `chk` (checkbox),
  `txt` (static text), etc. `ref` is the opaque id used by actions (`e3`).
- **Flags** are space-separated bare words or `key=value` pairs: `req`, `empty`, `disabled`,
  `primary`, `off`, `numeric`, `max=4`, `opts=243`, `="United States"` (current value),
  `x=14` (hidden/count), `more:e1` / `expand:e12` (a nested ref reachable from this line).
- Static text lines with no affordance use `txt` and have no ref unless actionable.
- No indentation is ever emitted; hierarchy is conveyed by `@sec`/`~` scope markers only.
