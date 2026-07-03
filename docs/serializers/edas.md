# EDAS — (spec pending)

> **Status:** ROADMAP / placeholder. EDAS was mentioned as a desired serializer but has no
> definition yet. **Action required:** obtain the EDAS grammar/definition from the user, then
> implement `modalitybench/observations/serializers/edas.py` following the same pattern as
> `afus.py` / `fct.py` (render from `PageGraph`, emit refs into the `RefRegistry`, register
> under the name `edas`).

Until the spec is provided, EDAS is intentionally **not** implemented, so runs that reference
it will fail fast with a clear "unknown strategy" error rather than silently using a stand-in.

## What we need to capture EDAS

To add EDAS, provide:
1. The line/record grammar (like AFUS's `TYPE:REF "name" [flags]` or FCT's pipe columns).
2. A worked example over a known page (ideally the same checkout-payment fixture used for
   AFUS/FCT) so it can be added to the golden-file tests.
3. How hierarchy/scope is represented, and how element refs are assigned.
