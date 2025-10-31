"""Shared test fixtures and configuration for pytest"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import SearchResults, VectorStore

# ============ Fixture Data ============


@pytest.fixture
def sample_course():
    """Sample course object for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction to ML",
                lesson_link="https://example.com/ml-course/lesson-0",
            ),
            Lesson(
                lesson_number=1,
                title="Supervised Learning Basics",
                lesson_link="https://example.com/ml-course/lesson-1",
            ),
            Lesson(
                lesson_number=2,
                title="Unsupervised Learning Methods",
                lesson_link="https://example.com/ml-course/lesson-2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Lesson 0 content: Machine learning is a subset of artificial intelligence.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="Machine learning has applications in various fields including computer vision.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Supervised learning is one of the most common types of machine learning.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results from vector store"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Supervised learning is one of the most common types.",
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 0},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
        ],
        distances=[0.15, 0.23],
    )


# ============ Mock Components ============


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Mock VectorStore with pre-configured responses"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search.return_value = sample_search_results
    mock_store._resolve_course_name.return_value = "Introduction to Machine Learning"
    mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson-0"

    # Mock course catalog get method
    mock_store.course_catalog = Mock()
    mock_store.course_catalog.get.return_value = {
        "metadatas": [
            {
                "title": "Introduction to Machine Learning",
                "course_link": "https://example.com/ml-course",
                "instructor": "Dr. Jane Smith",
                "lessons_json": '[{"lesson_number": 0, "lesson_title": "Introduction to ML", "lesson_link": "https://example.com/ml-course/lesson-0"}]',
                "lesson_count": 3,
            }
        ]
    }

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client"""
    mock_client = Mock()

    # Create mock response for direct queries (no tools)
    mock_direct_response = Mock()
    mock_direct_response.content = [Mock(text="This is a direct answer")]
    mock_direct_response.stop_reason = "end_turn"

    # Create mock response for tool use
    mock_tool_response = Mock()
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "machine learning"}
    mock_tool_response.content = [mock_tool_block]
    mock_tool_response.stop_reason = "tool_use"

    # Create mock final response after tool use
    mock_final_response = Mock()
    mock_final_response.content = [
        Mock(text="Based on the course materials, machine learning is...")
    ]
    mock_final_response.stop_reason = "end_turn"

    # Configure mock to return different responses
    mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

    return mock_client


@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    return Config(
        ANTHROPIC_API_KEY="test_key_123",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db",
    )


# ============ Component Fixtures ============


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance with mocked vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool instance with mocked vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def ai_generator_with_mock(mock_anthropic_client, test_config):
    """AIGenerator with mocked Anthropic client"""
    generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
    generator.client = mock_anthropic_client
    return generator


@pytest.fixture
def session_manager():
    """SessionManager instance"""
    return SessionManager(max_history=2)


# ============ File Path Fixtures ============


@pytest.fixture
def sample_course_file():
    """Path to sample course fixture file"""
    return Path(__file__).parent / "fixtures" / "sample_course.txt"


# ============ Cleanup Fixtures ============


@pytest.fixture(autouse=True)
def cleanup_test_db():
    """Clean up test database after each test"""
    yield
    test_db_path = Path("./test_chroma_db")
    if test_db_path.exists():
        import shutil

        shutil.rmtree(test_db_path, ignore_errors=True)
