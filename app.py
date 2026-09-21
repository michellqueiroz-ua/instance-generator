"""Compatibility shim: `streamlit run app.py` from a repository checkout.

The interface itself now lives in REQreate/webapp/app.py so that it ships
inside the installed package and can be launched with `reqreate app`. This
file keeps the old command (and the Docker/Hugging Face deployment, which
runs `streamlit run app.py`) working from a checkout.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REQreate", "webapp", "app.py")
sys.path.insert(0, os.path.dirname(_app))
runpy.run_path(_app, run_name="__main__")
