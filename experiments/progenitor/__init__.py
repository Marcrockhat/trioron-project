"""Progenitor-council rebuild — clean room (EXPERIMENT / driver layer).

A from-scratch, step-by-step build of the progenitor-council architecture
(docs/design/progenitor_council.md), on the bare trioron Arena (nodes + edges).
Built and checked one piece at a time; progenitor_council.md is the design
reference.

**This is the experiment layer — it does NOT ship.** Drivers here observe
(print, eval, plot) and import the architecture from the package. As each
mechanism (genesis, council, …) validates, the *pure* version migrates into
``trioron.progenitor`` (the API surface); these step scripts stay as the runnable
record of how it was built. The dependency direction is one-way: experiment →
package, never the reverse.
"""
