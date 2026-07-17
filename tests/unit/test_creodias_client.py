"""Unit tests for CREODIAS client/auth helpers (fully mocked)."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from tests.conftest import TEST_END, TEST_START

@pytest.mark.unit
def test_download_with_retry_succeeds_after_transient_failures():
    """_download_with_retry should retry through transient network
    exceptions and succeed once the underlying download() call stops
    failing, without needing the caller to handle anything."""
    import requests
    from HydroEO.downloaders import creodias

    call_count = {"n": 0}

    def fake_download(uid, token, outfile, show_progress=True):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise requests.exceptions.ChunkedEncodingError("dropped connection")
        # third attempt succeeds

    with (
        patch.object(creodias, "download", side_effect=fake_download),
        patch.object(creodias.time, "sleep") as mock_sleep,
    ):
        creodias._download_with_retry("uid123", token="tok", outfile="/tmp/x.zip")

    assert call_count["n"] == 3
    assert mock_sleep.call_count == 2  # backoff between attempts 1->2 and 2->3


@pytest.mark.unit
def test_download_with_retry_raises_after_exhausting_attempts():
    """A persistently-failing download should raise once retries are
    exhausted, so the caller (download_list) can decide to skip it."""
    import requests
    from HydroEO.downloaders import creodias

    with (
        patch.object(
            creodias,
            "download",
            side_effect=requests.exceptions.ConnectionError("still down"),
        ),
        patch.object(creodias.time, "sleep"),
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            creodias._download_with_retry(
                "uid123", token="tok", outfile="/tmp/x.zip", max_retries=3
            )


@pytest.mark.unit
def test_download_list_skips_persistently_failing_file_and_continues(tmp_path, caplog):
    """download_list must not let one file's persistent network failure
    crash the whole batch -- it should log the failure, skip that file,
    and continue downloading the rest."""
    import logging
    import requests
    from HydroEO.downloaders import creodias

    good_uids = ["good1", "good2"]
    bad_uid = "bad1"
    uids = ["good1", "bad1", "good2"]

    def fake_download(uid, token, outfile, show_progress=True):
        if uid == bad_uid:
            raise requests.exceptions.ChunkedEncodingError("dropped connection")
        Path(outfile).write_text("data")

    with (
        patch.object(creodias, "download", side_effect=fake_download),
        patch.object(creodias.time, "sleep"),
        patch.object(creodias, "_get_token", return_value="tok"),
        caplog.at_level(logging.WARNING),
    ):
        result_token, _ = creodias.download_list(
            uids,
            username="u",
            password="p",
            token="tok",
            session_start_time=__import__("time").time(),
            outdir=str(tmp_path),
            show_progress=False,
        )

    for uid in good_uids:
        assert (tmp_path / f"{uid}.zip").exists(), f"{uid} should have downloaded fine"
    assert not (tmp_path / f"{bad_uid}.zip").exists()
    assert "Giving up on bad1" in caplog.text
    assert "1 of 3 file(s) could not be downloaded" in caplog.text



@pytest.mark.unit
def test_creodias_get_token_raises_on_bad_credentials():
    """_get_token() must raise RuntimeError when the token endpoint returns no access_token."""
    from HydroEO.downloaders.creodias import _get_token

    bad_response = MagicMock()
    bad_response.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Invalid credentials",
    }

    with patch("HydroEO.downloaders.creodias.requests.post", return_value=bad_response):
        with pytest.raises(RuntimeError, match="Unable to get token"):
            _get_token("wrong_user", "wrong_pass")


@pytest.mark.unit
def test_creodias_query_raises_on_http_error():
    """creodias.query() must surface HTTP errors raised by requests."""
    import requests
    from HydroEO.downloaders.creodias import query

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

    with patch("HydroEO.downloaders.creodias.requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            query(
                collection="Sentinel3",
                start_date=TEST_START,
                end_date=TEST_END,
            )


@pytest.mark.unit
def test_parse_geometry_accepts_coordinate_tuple_list():
    """_parse_geometry() should accept a list of (lon, lat) tuples."""
    from HydroEO.downloaders.creodias import _parse_geometry

    geom = [
        (-70.5, -16.5),
        (-68.5, -16.5),
        (-68.5, -15.0),
        (-70.5, -15.0),
        (-70.5, -16.5),
    ]

    wkt = _parse_geometry(geom)

    assert wkt.startswith("POLYGON")


@pytest.mark.unit
def test_parse_geometry_accepts_flat_coordinate_list():
    """_parse_geometry() should accept a flat [lon, lat, ...] coordinate list."""
    from HydroEO.downloaders.creodias import _parse_geometry

    geom = [-70.5, -16.5, -68.5, -16.5, -68.5, -15.0, -70.5, -15.0, -70.5, -16.5]

    wkt = _parse_geometry(geom)

    assert wkt.startswith("POLYGON")
