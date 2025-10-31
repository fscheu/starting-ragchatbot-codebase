"""API endpoint tests for FastAPI application"""
import pytest
from unittest.mock import Mock
from httpx import AsyncClient


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for /api/query endpoint"""

    async def test_query_endpoint_success(self, test_app, test_client):
        """Test successful query with response"""
        # Configure mock RAG system
        test_app.state.mock_rag.query.return_value = (
            "Machine learning is a subset of artificial intelligence.",
            [{"text": "ML Course - Lesson 0", "link": "https://example.com/lesson-0"}]
        )

        # Make request
        response = await test_client.post(
            "/api/query",
            json={
                "query": "What is machine learning?",
                "session_id": "test_session_1"
            }
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        assert data["answer"] == "Machine learning is a subset of artificial intelligence."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "ML Course - Lesson 0"
        assert data["sources"][0]["link"] == "https://example.com/lesson-0"
        assert data["session_id"] == "test_session_1"

        # Verify RAG system was called correctly
        test_app.state.mock_rag.query.assert_called_once_with(
            "What is machine learning?",
            "test_session_1"
        )

    async def test_query_endpoint_without_session_id(self, test_app, test_client):
        """Test query endpoint creates session when not provided"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = (
            "Answer without session",
            []
        )

        # Make request without session_id
        response = await test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Should have generated a session_id
        assert "session_id" in data
        assert data["session_id"] == "test_session_1"

    async def test_query_endpoint_with_multiple_sources(self, test_app, test_client):
        """Test query endpoint with multiple source citations"""
        # Configure mock with multiple sources
        test_app.state.mock_rag.query.return_value = (
            "Machine learning includes supervised and unsupervised learning.",
            [
                {"text": "ML Course - Lesson 0", "link": "https://example.com/lesson-0"},
                {"text": "ML Course - Lesson 1", "link": "https://example.com/lesson-1"},
                {"text": "ML Course - Lesson 2", "link": "https://example.com/lesson-2"}
            ]
        )

        # Make request
        response = await test_client.post(
            "/api/query",
            json={"query": "Types of machine learning"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert len(data["sources"]) == 3
        for i, source in enumerate(data["sources"]):
            assert "text" in source
            assert "link" in source
            assert f"Lesson {i}" in source["text"]

    async def test_query_endpoint_with_sources_without_links(self, test_app, test_client):
        """Test query endpoint with sources that have no links"""
        # Configure mock with sources without links
        test_app.state.mock_rag.query.return_value = (
            "Answer from general content",
            [
                {"text": "General reference", "link": None},
                {"text": "Another reference", "link": None}
            ]
        )

        # Make request
        response = await test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert len(data["sources"]) == 2
        for source in data["sources"]:
            assert source["link"] is None

    async def test_query_endpoint_empty_query(self, test_app, test_client):
        """Test query endpoint with empty query string"""
        # Configure mock to handle empty query
        test_app.state.mock_rag.query.return_value = (
            "Please provide a question.",
            []
        )

        # Make request with empty query
        response = await test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Should still succeed (validation happens in RAG system)
        assert response.status_code == 200

    async def test_query_endpoint_error_handling(self, test_app, test_client):
        """Test query endpoint error handling"""
        # Configure mock to raise exception
        test_app.state.mock_rag.query.side_effect = Exception("Database connection error")

        # Make request
        response = await test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection error" in data["detail"]

    async def test_query_endpoint_invalid_request_format(self, test_client):
        """Test query endpoint with invalid request format"""
        # Make request with missing required field
        response = await test_client.post(
            "/api/query",
            json={"session_id": "test"}  # Missing 'query' field
        )

        # Should return validation error
        assert response.status_code == 422

    async def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        # Make request with invalid JSON
        response = await test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return error
        assert response.status_code == 422

    async def test_query_endpoint_no_sources(self, test_app, test_client):
        """Test query endpoint when no sources are returned"""
        # Configure mock with empty sources
        test_app.state.mock_rag.query.return_value = (
            "This is a general knowledge answer.",
            []
        )

        # Make request
        response = await test_client.post(
            "/api/query",
            json={"query": "What is 2+2?"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "This is a general knowledge answer."
        assert data["sources"] == []
        assert "session_id" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for /api/courses endpoint"""

    async def test_courses_endpoint_success(self, test_app, test_client):
        """Test successful retrieval of course statistics"""
        # Configure mock
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "Introduction to Machine Learning",
                "Deep Learning Fundamentals",
                "Natural Language Processing"
            ]
        }

        # Make request
        response = await test_client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Machine Learning" in data["course_titles"]

    async def test_courses_endpoint_no_courses(self, test_app, test_client):
        """Test courses endpoint when no courses exist"""
        # Configure mock with empty courses
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Make request
        response = await test_client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    async def test_courses_endpoint_many_courses(self, test_app, test_client):
        """Test courses endpoint with many courses"""
        # Configure mock with many courses
        course_titles = [f"Course {i}" for i in range(50)]
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 50,
            "course_titles": course_titles
        }

        # Make request
        response = await test_client.get("/api/courses")

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 50
        assert len(data["course_titles"]) == 50

    async def test_courses_endpoint_error_handling(self, test_app, test_client):
        """Test courses endpoint error handling"""
        # Configure mock to raise exception
        test_app.state.mock_rag.get_course_analytics.side_effect = Exception(
            "Vector store error"
        )

        # Make request
        response = await test_client.get("/api/courses")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Vector store error" in data["detail"]


@pytest.mark.api
class TestCORSMiddleware:
    """Tests for CORS middleware configuration"""

    async def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in response"""
        # Make preflight request
        response = await test_client.options(
            "/api/courses",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Verify CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    async def test_cors_allows_post_requests(self, test_app, test_client):
        """Test that CORS allows POST requests"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # Make POST request with origin header
        response = await test_client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Origin": "http://localhost:3000"}
        )

        # Should succeed with CORS headers
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


@pytest.mark.api
class TestRequestValidation:
    """Tests for request validation and data types"""

    async def test_query_with_wrong_data_types(self, test_client):
        """Test query endpoint with wrong data types"""
        # Make request with integer instead of string
        response = await test_client.post(
            "/api/query",
            json={"query": 123}  # Should be string
        )

        # Should return validation error
        assert response.status_code == 422

    async def test_query_with_extra_fields(self, test_app, test_client):
        """Test query endpoint ignores extra fields"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # Make request with extra fields
        response = await test_client.post(
            "/api/query",
            json={
                "query": "Test",
                "extra_field": "ignored",
                "another_field": 123
            }
        )

        # Should succeed (extra fields ignored by Pydantic)
        assert response.status_code == 200

    async def test_query_with_very_long_string(self, test_app, test_client):
        """Test query endpoint with very long query string"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # Make request with very long query
        long_query = "What is machine learning? " * 1000
        response = await test_client.post(
            "/api/query",
            json={"query": long_query}
        )

        # Should succeed (no length limit in model)
        assert response.status_code == 200

    async def test_query_with_special_characters(self, test_app, test_client):
        """Test query endpoint with special characters"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # Make request with special characters
        response = await test_client.post(
            "/api/query",
            json={"query": "What is ML? 你好 🤖 <script>alert('test')</script>"}
        )

        # Should succeed
        assert response.status_code == 200
        assert test_app.state.mock_rag.query.called


@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests for multiple endpoint interactions"""

    async def test_query_then_courses(self, test_app, test_client):
        """Test querying then getting course stats"""
        # Configure mocks
        test_app.state.mock_rag.query.return_value = ("Answer", [])
        test_app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["ML Course"]
        }

        # Make query request
        query_response = await test_client.post(
            "/api/query",
            json={"query": "Test"}
        )
        assert query_response.status_code == 200

        # Make courses request
        courses_response = await test_client.get("/api/courses")
        assert courses_response.status_code == 200

    async def test_multiple_queries_same_session(self, test_app, test_client):
        """Test multiple queries with the same session ID"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # First query
        response1 = await test_client.post(
            "/api/query",
            json={"query": "First question", "session_id": "session_123"}
        )
        assert response1.status_code == 200
        assert response1.json()["session_id"] == "session_123"

        # Second query with same session
        response2 = await test_client.post(
            "/api/query",
            json={"query": "Second question", "session_id": "session_123"}
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == "session_123"

        # Verify both calls were made
        assert test_app.state.mock_rag.query.call_count == 2

    async def test_concurrent_sessions(self, test_app, test_client):
        """Test handling multiple concurrent sessions"""
        # Configure mock
        test_app.state.mock_rag.query.return_value = ("Answer", [])

        # Make requests with different sessions
        response1 = await test_client.post(
            "/api/query",
            json={"query": "Query A", "session_id": "session_A"}
        )
        response2 = await test_client.post(
            "/api/query",
            json={"query": "Query B", "session_id": "session_B"}
        )
        response3 = await test_client.post(
            "/api/query",
            json={"query": "Query C", "session_id": "session_C"}
        )

        # All should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200

        # Each should maintain their session ID
        assert response1.json()["session_id"] == "session_A"
        assert response2.json()["session_id"] == "session_B"
        assert response3.json()["session_id"] == "session_C"
