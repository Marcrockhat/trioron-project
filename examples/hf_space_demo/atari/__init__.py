"""Atari Showcase tab — runs trioron organisms on Pong / Breakout
and records video for the HF Space demo.

Each `play_match` invocation runs one full ALE episode end-to-end
(server-side), records gameplay to MP4, returns the MP4 path along
with summary stats. The Gradio UI in `showcase.py` wires these into
a 3-panel comparison layout with running win-rate accumulation.
"""
