"""Progenitor–council mechanism — the API surface (architecture).

This subpackage is the *shipped* home for the progenitor–council architecture
(design: ``docs/design/progenitor_council.md``): input-layer genesis (oversized
aperture → variance apoptosis → saliency), the standing 5×4 council, and
trial-vote differentiation.

**Migration policy (set session 025).** Code lands here only once it is validated
in the experiment layer (``experiments/progenitor/``). Each mechanism is a *pure*
function/class — no printing, no eval harness, no testbed data — so it can be
wrapped directly as an API. The experiment scripts import from here; this package
never imports from ``experiments``.

Currently empty: the genesis and council mechanisms are still being validated
(the dendrite-wins-the-vote hypothesis was not confirmed at step 3b). They migrate
in as each piece is proven. The genuine phenotype ops they rely on already live in
``trioron.phenotype`` (linear, attention, conv, recurrent, dendrite).
"""
