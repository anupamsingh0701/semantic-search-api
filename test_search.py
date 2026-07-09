import os
import unittest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from main import app, cosine_similarity

class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_cosine_similarity_orthogonal(self):
        # Orthogonal vectors should have 0 similarity
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)

    def test_cosine_similarity_identical(self):
        # Identical vectors should have 1 similarity
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

    def test_cosine_similarity_opposite(self):
        # Opposite vectors should have -1 similarity
        v1 = [1.0, -1.0]
        v2 = [-1.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), -1.0)

    def test_cosine_similarity_zero_vector(self):
        # Handle zero vector gracefully
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        self.assertEqual(cosine_similarity(v1, v2), 0.0)

    @patch("main.get_embeddings", new_callable=AsyncMock)
    @patch.dict(os.environ, {"AIPIPE_TOKEN": "mock-token-123"})
    def test_rank_candidates_success(self, mock_get_embeddings):
        # We mock 4 embeddings (1 query + 3 candidates)
        # Query: [1, 0, 0]
        # Candidate 0: [0, 1, 0] -> Sim = 0
        # Candidate 1: [1, 0, 0] -> Sim = 1 (most similar)
        # Candidate 2: [0.707, 0.707, 0] -> Sim = 0.707 (second most similar)
        mock_get_embeddings.return_value = [
            [1.0, 0.0, 0.0],  # Query
            [0.0, 1.0, 0.0],  # Candidate 0
            [1.0, 0.0, 0.0],  # Candidate 1
            [0.70710678, 0.70710678, 0.0]  # Candidate 2
        ]

        payload = {
            "query_id": "qtest",
            "query": "find matching",
            "candidates": [
                "orthogonal candidate",
                "perfect match candidate",
                "partial match candidate"
            ]
        }

        # Test both / and /rank
        for path in ["/", "/rank", "/search"]:
            response = self.client.post(path, json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("ranking", data)
            # The order should be Candidate 1 (sim=1.0), Candidate 2 (sim=0.707), Candidate 0 (sim=0.0)
            self.assertEqual(data["ranking"], [1, 2, 0])

        mock_get_embeddings.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        # Should raise 500 when no API keys are present in env
        payload = {
            "query_id": "qtest",
            "query": "find matching",
            "candidates": ["some candidate"]
        }
        response = self.client.post("/rank", json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertIn("API Key/Token not configured", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
