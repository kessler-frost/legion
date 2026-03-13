import pytest
from httpx import ASGITransport, AsyncClient

from brain.api.main import app


@pytest.mark.asyncio
async def test_post_command_returns_200():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/bot/1/command",
            json={"action": "forward", "params": {"speed": 1500}},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "sent"


@pytest.mark.asyncio
async def test_post_command_invalid_action_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/bot/1/command",
            json={"action": "fly", "params": {}},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_root_serves_html():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Legion" in response.text
