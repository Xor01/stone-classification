def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api"] == "healthy"
    assert "database" in body
    assert "model" in body
