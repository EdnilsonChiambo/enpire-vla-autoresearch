from __future__ import annotations

import json
import os
import time

DEFAULT_LOG_PATH = "/Users/ednilsonchiambo/enpire-vla-autoresearch/.cursor/debug-b95471.log"
SESSION_ID = "b95471"


def debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    entry = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    log_path = os.environ.get("DEBUG_LOG_PATH", DEFAULT_LOG_PATH)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    print(f"DEBUG_LOG {json.dumps(entry)}")
    # #endregion
