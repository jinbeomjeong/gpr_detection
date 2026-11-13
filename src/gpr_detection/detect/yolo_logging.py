"""Blind-run log level for Ultralytics (``--blind``, M3-1 N2-a plan section 4-2).

Ultralytics 8.4.192 prints the per-epoch validation table and the evaluation
summary through ``ultralytics.utils.LOGGER`` at INFO. Its progress bars turn
themselves off when that logger is above INFO (``utils/tqdm.py``), so the
validation bar header ("... mAP50 mAP50-95") disappears too.

``ultralytics`` is imported only inside the function (ADR-001). Importing it
first matters: its import sets the logger level, so a level set earlier would
be overwritten.
"""

import logging

BLIND_LOG_LEVEL = logging.WARNING


def quiet_ultralytics_logger(level: int = BLIND_LOG_LEVEL) -> None:
    """Import Ultralytics' logger and raise its level (default WARNING).

    Affects this process only; spawned DataLoader workers keep INFO, but they
    log no metrics.
    """
    from ultralytics.utils import LOGGER  # lazy (ADR-001)

    LOGGER.setLevel(level)
    for handler in LOGGER.handlers:
        handler.setLevel(level)
