from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from cotton_doctor.web import create_app


class StubPredictor:
    def __call__(self, image):
        return {
            "detections": [{"class_id": 2, "class_name": "Armyworm", "confidence": 0.91,
                            "xyxy": [1.0, 2.0, 9.0, 10.0]}],
            "inference_ms": 4.2,
        }, b"jpeg-result"


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), (20, 80, 30)).save(output, format="JPEG")
    return output.getvalue()


def test_predict_returns_detections_and_cached_visualization():
    client = TestClient(create_app(StubPredictor()))

    response = client.post("/api/predict", files={"file": ("cotton.jpg", _jpeg(), "image/jpeg")})

    assert response.status_code == 200
    result = response.json()
    assert result["detections"][0]["class_name"] == "Armyworm"
    assert result["inference_ms"] == 4.2
    image_response = client.get(f"/api/results/{result['result_id']}.jpg")
    assert image_response.status_code == 200
    assert image_response.content == b"jpeg-result"


def test_predict_rejects_non_image_content_type():
    client = TestClient(create_app(StubPredictor()))

    response = client.post("/api/predict", files={"file": ("notes.txt", b"hi", "text/plain")})

    assert response.status_code == 415


def test_predict_rejects_unreadable_image():
    client = TestClient(create_app(StubPredictor()))

    response = client.post("/api/predict", files={"file": ("bad.jpg", b"not-jpeg", "image/jpeg")})

    assert response.status_code == 400
