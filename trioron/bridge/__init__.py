"""Cross-modal bridge for trioron.

Encoders translate raw input (text, image, audio) into a fixed-dim
feature vector that lands in trioron's shared L0 code space. Tools
turn trioron's output into structured external actions.

See `trioron.bridge.encoders` for reference encoder implementations
and `trioron.bridge.tools` for the JSON-schema and decorator-based
tool dispatchers.
"""
