import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24000/1001:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3:sample_rate=48000",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(p),
        ],
        check=True,
    )
    return p
