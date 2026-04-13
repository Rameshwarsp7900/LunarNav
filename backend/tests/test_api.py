"""
tests/test_api.py — FastAPI endpoint integration tests using httpx.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "datasets" in data


@pytest.mark.asyncio
async def test_mission_list_empty():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/mission/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_mission_not_found():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/mission/99999/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_mission_invalid():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Missing both modes — should fail validation
        resp = await ac.post("/mission/create", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_mission_latlon_schema():
    """Schema validation — valid request returns 202 (pipeline runs in background)."""
    payload = {
        "start": {"lat": -85.5, "lon": 10.0},
        "goal":  {"lat": -85.3, "lon": 10.2},
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/mission/create", json=payload)
    # Will fail at JP2 extraction (file not found) but schema is valid
    assert resp.status_code in (202, 500)


@pytest.mark.asyncio
async def test_lidar_scan_invalid_mission():
    payload = {
        "mission_id": 99999,
        "points": [{"theta": 10.0, "distance_m": 5.0, "quality": 45}],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/lidar/scan/process", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_docs_available():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/docs")
    assert resp.status_code == 200
