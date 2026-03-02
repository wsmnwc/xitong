"""后端模型与工具的单元测试"""

import json
import numpy as np
import pytest
from PIL import Image

from backend.config import CLIP_FEATURE_DIM, ALGORITHM_CF_DMW, ALGORITHM_CF_DF
from backend.models.base import DetectionResult
from backend.models.cf_dmw import CfDmwModel
from backend.models.cf_df import CfDfModel
from backend.utils.preprocessing import (
    clean_text,
    create_placeholder_image,
    load_image,
    parse_batch_csv,
    parse_batch_json,
)
from backend.utils.feature_extraction import FeatureExtractor


# ===== 预处理测试 =====


class TestPreprocessing:
    def test_clean_text_strips_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_clean_text_collapses_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_clean_text_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_create_placeholder_image(self):
        img = create_placeholder_image()
        assert isinstance(img, Image.Image)
        assert img.size == (224, 224)
        assert img.mode == "RGB"

    def test_load_image_from_pil(self):
        img = Image.new("RGBA", (100, 100), "red")
        result = load_image(img)
        assert result.mode == "RGB"

    def test_load_image_from_numpy(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = load_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_parse_batch_csv(self):
        csv_content = "text,image\nhello world,\ntest text,\n"
        items = parse_batch_csv(csv_content)
        assert len(items) == 2
        assert items[0]["text"] == "hello world"
        assert items[1]["text"] == "test text"

    def test_parse_batch_json(self):
        data = [
            {"text": "hello world"},
            {"text": "test text", "image": None},
        ]
        items = parse_batch_json(json.dumps(data))
        assert len(items) == 2
        assert items[0]["text"] == "hello world"

    def test_parse_batch_json_invalid(self):
        with pytest.raises(ValueError):
            parse_batch_json('{"not": "a list"}')


# ===== 特征提取测试 =====


class TestFeatureExtraction:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    def test_text_features_shape(self, extractor):
        features = extractor.extract_text_features("hello world")
        assert features.shape == (CLIP_FEATURE_DIM,)

    def test_text_features_normalized(self, extractor):
        features = extractor.extract_text_features("test text")
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_image_features_shape(self, extractor):
        img = create_placeholder_image()
        features = extractor.extract_image_features(img)
        assert features.shape == (CLIP_FEATURE_DIM,)

    def test_image_features_normalized(self, extractor):
        img = Image.new("RGB", (100, 100), "blue")
        features = extractor.extract_image_features(img)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_text_features_deterministic(self, extractor):
        f1 = extractor.extract_text_features("same text")
        f2 = extractor.extract_text_features("same text")
        np.testing.assert_array_almost_equal(f1, f2)

    def test_different_texts_different_features(self, extractor):
        f1 = extractor.extract_text_features("hello")
        f2 = extractor.extract_text_features("world")
        assert not np.allclose(f1, f2)


# ===== CF-DMW 模型测试 =====


class TestCfDmwModel:
    @pytest.fixture
    def model(self):
        return CfDmwModel(device="cpu")

    def test_predict_returns_detection_result(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert isinstance(result, DetectionResult)

    def test_predict_probability_in_range(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert 0.0 <= result.hate_probability <= 1.0

    def test_weights_sum_to_one(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert abs(result.text_weight + result.image_weight - 1.0) < 0.01

    def test_extra_contains_algorithm(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert result.extra is not None
        assert result.extra["algorithm"] == "cf-dmw"
        assert "dynamic_weights" in result.extra

    def test_algorithm_name(self, model):
        name = model.get_algorithm_name()
        assert "CF-DMW" in name


# ===== CF-DF 模型测试 =====


class TestCfDfModel:
    @pytest.fixture
    def model(self):
        return CfDfModel(device="cpu")

    def test_predict_returns_detection_result(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert isinstance(result, DetectionResult)

    def test_predict_probability_in_range(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert 0.0 <= result.hate_probability <= 1.0

    def test_extra_contains_algorithm(self, model):
        text_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        img_feat = np.random.randn(CLIP_FEATURE_DIM).astype(np.float32)
        result = model.predict(text_feat, img_feat)
        assert result.extra is not None
        assert result.extra["algorithm"] == "cf-df"
        assert "raw_probability" in result.extra
        assert "purified_probability" in result.extra

    def test_algorithm_name(self, model):
        name = model.get_algorithm_name()
        assert "CF-DF" in name


# ===== FastAPI 接口测试 =====


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.app import app

        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert ALGORITHM_CF_DMW in data["available_algorithms"]
        assert ALGORITHM_CF_DF in data["available_algorithms"]

    def test_detect_json_cf_dmw(self, client):
        response = client.post(
            "/detect/json",
            json={
                "text": "This is a test text",
                "algorithm": ALGORITHM_CF_DMW,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_hateful" in data
        assert "hate_probability" in data
        assert 0.0 <= data["hate_probability"] <= 1.0

    def test_detect_json_cf_df(self, client):
        response = client.post(
            "/detect/json",
            json={
                "text": "Another test text",
                "algorithm": ALGORITHM_CF_DF,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_hateful" in data
        assert data["extra"]["algorithm"] == "cf-df"

    def test_detect_form_upload(self, client):
        response = client.post(
            "/detect",
            data={"text": "test text", "algorithm": ALGORITHM_CF_DMW},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_hateful" in data

    def test_batch_csv(self, client):
        csv_content = "text\nhello world\ntest text\n"
        import io

        response = client.post(
            "/detect/batch",
            data={"algorithm": ALGORITHM_CF_DMW},
            files={"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_batch_json(self, client):
        json_content = json.dumps([
            {"text": "hello world"},
            {"text": "test text"},
        ])
        import io

        response = client.post(
            "/detect/batch",
            data={"algorithm": ALGORITHM_CF_DF},
            files={
                "file": (
                    "test.json",
                    io.BytesIO(json_content.encode()),
                    "application/json",
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
