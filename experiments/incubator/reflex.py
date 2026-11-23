"""The reflex brain: frozen Qwen2.5-1.5B (GGUF, llama_cpp), world-side only.

One pass = the reflex's short answer, its own confidence (per-token log-probs
of the greedy answer) and its final-layer hidden state at two points: the end
of the prompt (before it answers) and the end of the answer.  A single llama
context with embeddings switched on returns logits AND the last token's hidden
state from the same decode (verified s064: logits bit-identical to plain mode).
Do not use llama-cpp-python's `embedding=True` constructor flag: it inflates
the context to 256 sequences (3.7 GB) and breaks ordinary decoding.

See docs/design/incubator_matron.md §1.
"""
from __future__ import annotations
import os
import numpy as np
import llama_cpp
from llama_cpp import Llama

MODEL = os.environ.get(
    "REFLEX_GGUF",
    os.path.expanduser("~/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
SYSTEM = "Answer with a short phrase only."


def chat_prompt(question: str, system: str = SYSTEM) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


@llama_cpp.llama_log_callback
def _quiet(level, text, user_data):          # llama.cpp logs one line per decode otherwise
    pass


class Reflex:
    def __init__(self, n_ctx: int = 512, n_threads: int = 6):
        llama_cpp.llama_log_set(_quiet, None)
        self.m = Llama(MODEL, n_ctx=n_ctx, n_threads=n_threads, n_threads_batch=n_threads,
                       verbose=False)
        self.ctx = self.m._ctx.ctx
        llama_cpp.llama_set_embeddings(self.ctx, True)
        self.V, self.d = self.m.n_vocab(), self.m.n_embd()
        self._head_kept = False                 # read() keeps the prompt head decoded
        self.eos = {self.m.token_eos(),
                    self.m.tokenize(b"<|im_end|>", add_bos=False, special=True)[0]}

    def tokens(self, text: str) -> list[int]:
        return self.m.tokenize(text.encode(), add_bos=False, special=True)

    def _logp(self) -> np.ndarray:
        return _log_softmax(np.ctypeslib.as_array(
            llama_cpp.llama_get_logits_ith(self.ctx, -1), shape=(self.V,)))

    def _hidden(self) -> np.ndarray:
        return np.ctypeslib.as_array(
            llama_cpp.llama_get_embeddings_ith(self.ctx, -1), shape=(self.d,)).copy()

    def read(self, question: str) -> dict:
        """Read a question WITHOUT answering it (stage 2 keys): the hidden state
        at the prompt's last token (h_last) and the mean hidden state over the
        question's own tokens (h_mean).  The fixed prompt head is decoded once
        and kept; each call decodes only the question and the prompt tail, with
        every token flagged for output.  h_last matches ``answer``'s h_pre only
        to cos ~.999 (different batch chunking): compare keys from ``read`` only
        with keys from ``read``."""
        m = self.m
        head, tail = (self.tokens(t) for t in chat_prompt("\x00").split("\x00"))
        if not self._head_kept:
            m.reset()
            m._ctx.kv_cache_clear()
            m._batch.set_batch(head, 0, False)
            m._ctx.decode(m._batch)
            self._head_kept = True
        q = self.tokens(question)
        m._ctx.kv_cache_seq_rm(0, len(head), -1)
        m._batch.set_batch(q + tail, len(head), True)
        m._ctx.decode(m._batch)
        E = np.stack([np.ctypeslib.as_array(
            llama_cpp.llama_get_embeddings_ith(self.ctx, i), shape=(self.d,))
            for i in range(len(q) + len(tail))])
        return {"h_last": E[-1].copy(), "h_mean": E[:len(q)].mean(0), "n_q": len(q)}

    def answer(self, prompt: str, max_tokens: int = 24) -> dict:
        """Greedy decode.  Returns text, token log-probs, and the hidden state
        at the prompt's last token (h_pre) and the answer's last token (h_post)."""
        m = self.m
        ids = self.tokens(prompt)
        self._head_kept = False
        m.reset()
        m._ctx.kv_cache_clear()
        m.eval(ids)
        h_pre = self._hidden()
        h_post, out, lps = h_pre, [], []
        for _ in range(max_tokens):
            lp = self._logp()
            t = int(lp.argmax())
            if t in self.eos:
                break
            out.append(t); lps.append(float(lp[t]))
            m.eval([t])
            h_post = self._hidden()
        return {"text": m.detokenize(out).decode("utf-8", "ignore").strip(),
                "logprobs": lps, "n_prompt": len(ids), "h_pre": h_pre, "h_post": h_post}
