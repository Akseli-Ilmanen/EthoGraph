import re
import sys

from ethograph.utils.logging import start_session_log


def test_start_session_log_writes_timestamped_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHOGRAPH_HOME", str(tmp_path))
    real_stdout, real_stderr = sys.stdout, sys.stderr

    path = start_session_log("test")
    try:
        assert path.parent == tmp_path / "logs"
        assert path.exists()

        print("hello")
        print("multi\nline")
        sys.stdout.flush()
    finally:
        sys.stdout._log_file.close()
        sys.stdout, sys.stderr = real_stdout, real_stderr

    content = path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if not line.startswith("#")]
    assert len(lines) == 3
    for line in lines:
        assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ", line)
    assert lines[0].endswith("hello")
    assert lines[1].endswith("multi")
    assert lines[2].endswith("line")
