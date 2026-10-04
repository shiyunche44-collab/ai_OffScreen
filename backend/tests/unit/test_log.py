import json
import logging
from pathlib import Path

import pytest

from offscreen.log import bind_job, get_logger, setup_logging


def test_json_lines_with_job_context_and_scrubbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOME_KEY", "sk-very-secret-value")
    setup_logging(tmp_path, secret_env_names=["SOME_KEY"])
    log = get_logger("t")
    with bind_job("job_abc"):
        log.info("calling with key sk-very-secret-value now")
    log.warning("no job here")
    for h in logging.getLogger("offscreen").handlers:
        h.flush()
    lines = [
        json.loads(x) for x in (tmp_path / "offscreen.log").read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["job_id"] == "job_abc"
    assert "sk-very-secret-value" not in json.dumps(lines)
    assert "***" in lines[0]["msg"]
    assert "job_id" not in lines[1] and lines[1]["level"] == "warning"
