import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_api  # noqa: E402


@pytest.fixture(scope="session")
def api():
    server, url = fake_api.start()
    yield url
    server.shutdown()
