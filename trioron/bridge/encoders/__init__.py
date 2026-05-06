"""Reference encoder implementations for the trioron bridge.

Each encoder is gated behind its own optional dependency extra so the
core trioron package stays lean. Install bridge deps with one of:

    pip install trioron[bridge-text]   # sentence-transformers
    pip install trioron[bridge-image]  # open-clip-torch + Pillow
    pip install trioron[bridge-audio]  # openai-whisper
    pip install trioron[bridge-all]    # all three

Importing an encoder without its extras installed raises ImportError
with the exact pip command needed.
"""
