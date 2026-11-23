"""Organisms — saveable compositions of substrates with a uniform API (s062).

* :class:`Logophosis` — grounded-language comprehension (Phasecyte Link 0 on
  chord words + a growing substrate chain).
* :class:`Kinopsis` — motion-sight (shape organ + motion/velocity leaves +
  evidence router).
* :class:`Vee` — awareness on a reflex brain: predicts when a frozen
  language model's answer is wrong from its hidden state (s064, stage 1).
"""
from .base import API_VERSION, Organism, Std, chunked, leaf, train_leaf
from .kinopsis import Kinopsis, MotionOrganism, QDAVoter, Router, conf, vote
from .logophosis import ChordLink0, Logophosis, OracleLink0
from .vee import Vee

__all__ = ["API_VERSION", "Organism", "Std", "chunked", "leaf", "train_leaf",
           "Kinopsis", "MotionOrganism", "QDAVoter", "Router", "conf", "vote",
           "ChordLink0", "Logophosis", "OracleLink0", "Vee"]
