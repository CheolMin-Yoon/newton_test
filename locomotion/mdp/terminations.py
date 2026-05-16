"""Termination terms: 낙상(base 자세/높이) + time_out(mjlab mdp.time_out)."""

from __future__ import annotations


def base_fell(env):
  raise NotImplementedError("base roll/pitch 또는 height 임계")
