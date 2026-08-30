#!/usr/bin/env python
"""Launcher for speed_compare without modifying app.py.

1) Pre-imports real Gradio so repo-root `gradio/` cannot shadow the package.
2) Compatibility shim: app.py passes css= to gr.Blocks (Gradio<=5 style);
   Gradio 6 expects css= on launch() instead.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import gradio as gr

# Keep the real package even after app.py inserts the repo into sys.path.
sys.modules["gradio"] = gr

_orig_blocks_init = gr.Blocks.__init__
_orig_launch = gr.Blocks.launch


def _blocks_init(self, *args, css=None, **kwargs):
    self._speed_compare_css = css
    return _orig_blocks_init(self, *args, **kwargs)


def _blocks_launch(self, *args, **kwargs):
    css = getattr(self, "_speed_compare_css", None)
    if css is not None and "css" not in kwargs:
        kwargs["css"] = css
    return _orig_launch(self, *args, **kwargs)


gr.Blocks.__init__ = _blocks_init
gr.Blocks.launch = _blocks_launch

APP = REPO / "chat" / "speed_compare" / "app.py"
runpy.run_path(str(APP), run_name="__main__")
