from sqlmodel import Session, select

from app import database
from app.models import Prediction


def test_prediction_row_persists(client):
    from tests.test_predictions import _fake_image_bytes

    files = {"image": ("dog.jpg", _fake_image_bytes(), "image/jpeg")}
    client.post("/api/v1/predict", files=files)

    # Look up database.engine dynamically — the test fixture monkeypatches
    # this module attribute to point at a throwaway SQLite DB per test.
    with Session(database.engine) as session:
        rows = session.exec(select(Prediction)).all()
        assert len(rows) == 1
        assert rows[0].image_name == "dog.jpg"
        assert rows[0].request_id is not None
