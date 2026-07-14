"""Install LexBrowser's optional Nemo-Gym GRPO grouping adapter in Ray workers."""

import os

if os.environ.get("LEXBROWSER_STABLE_GRPO_GROUPING") == "1":
    try:
        from lexbrowser_nemo_gym_grouping import install

        install()
    except ModuleNotFoundError:
        # NeMo-Gym's isolated helper virtualenvs do not import NeMo-RL.  They
        # intentionally run without this driver/Ray-worker-only adapter.
        pass
