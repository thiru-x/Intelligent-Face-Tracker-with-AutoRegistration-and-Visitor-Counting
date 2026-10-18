import json
import threading
import urllib.error
import urllib.request

import pytest

from app.web import create_server


class FakeController:
    def __init__(self):
        self.started = None

    def snapshot(self):
        return {
            "state": "idle",
            "has_frame": False,
            "recent_events": [],
            "database_unique_identities": 0,
            "database_visits": 0,
        }

    def start(self, source=None, camera=None):
        self.started = (source, camera)

    def stop(self):
        return False


@pytest.fixture
def dashboard_server():
    controller = FakeController()
    server = create_server(
        host="127.0.0.1",
        port=0,
        controller=controller,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, controller
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_dashboard_serves_page_and_status_api(dashboard_server):
    server, _ = dashboard_server
    base_url = f"http://127.0.0.1:{server.server_port}"

    with urllib.request.urlopen(base_url + "/") as response:
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert "Local Visitor Monitor" in page
        assert "/video.mjpg" in page

    with urllib.request.urlopen(base_url + "/api/status") as response:
        status = json.loads(response.read())

    assert status["state"] == "idle"
    assert status["recent_events"] == []


def test_dashboard_start_api_accepts_local_video_and_camera(
    dashboard_server,
):
    server, controller = dashboard_server
    base_url = f"http://127.0.0.1:{server.server_port}"

    for payload, expected in (
        ({"source": "sample_video.mp4"}, ("sample_video.mp4", None)),
        (
            {"source": "rtsp://camera.local:554/stream1"},
            ("rtsp://camera.local:554/stream1", None),
        ),
        (
            {"source": "rtsps://camera.local:322/stream1"},
            ("rtsps://camera.local:322/stream1", None),
        ),
        ({"camera": 0}, (None, 0)),
    ):
        request = urllib.request.Request(
            base_url + "/api/start",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
            assert json.loads(response.read())["state"] == "starting"
        assert controller.started == expected


def test_dashboard_refuses_non_loopback_binding():
    with pytest.raises(ValueError, match="localhost"):
        create_server(host="0.0.0.0", port=8000)


def test_dashboard_rejects_malformed_start_request(dashboard_server):
    server, controller = dashboard_server
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/api/start",
        data=b'{"camera": -1}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)

    assert error.value.code == 400
    assert controller.started is None


def test_dashboard_rejects_non_local_host_header(dashboard_server):
    server, _ = dashboard_server
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/api/status",
        headers={"Host": "attacker.example"},
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)

    assert error.value.code == 403
