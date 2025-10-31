"""Unit tests for CourseSearchTool"""
import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_basic_search(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test basic search with query only"""
        # Execute search
        result = course_search_tool.execute(query="machine learning")

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains expected content
        assert "Machine learning is a subset" in result
        assert "[Introduction to Machine Learning" in result

    def test_execute_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test search with course name filter"""
        # Execute search with course filter
        result = course_search_tool.execute(
            query="supervised learning",
            course_name="Machine Learning"
        )

        # Verify vector store search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="supervised learning",
            course_name="Machine Learning",
            lesson_number=None
        )

        # Should return formatted results
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        # Execute search with lesson filter
        result = course_search_tool.execute(
            query="introduction",
            lesson_number=0
        )

        # Verify vector store search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name=None,
            lesson_number=0
        )

        # Should return formatted results
        assert isinstance(result, str)

    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test search with both course name and lesson number filters"""
        # Execute search with both filters
        result = course_search_tool.execute(
            query="algorithms",
            course_name="ML Course",
            lesson_number=1
        )

        # Verify both filters were passed to vector store
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name="ML Course",
            lesson_number=1
        )

    def test_execute_no_results(self, course_search_tool, mock_vector_store):
        """Test search that returns no results"""
        # Configure mock to return empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        # Execute search
        result = course_search_tool.execute(query="nonexistent topic")

        # Should return "No relevant content found" message
        assert "No relevant content found" in result

    def test_execute_no_results_with_filters(self, course_search_tool, mock_vector_store):
        """Test search with filters that returns no results"""
        # Configure mock to return empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        # Execute search with filters
        result = course_search_tool.execute(
            query="topic",
            course_name="Nonexistent Course",
            lesson_number=99
        )

        # Should include filter information in message
        assert "No relevant content found" in result
        assert "Nonexistent Course" in result
        assert "lesson 99" in result

    def test_execute_invalid_course(self, course_search_tool, mock_vector_store):
        """Test search with non-existent course name"""
        # Configure mock to return error for invalid course
        error_results = SearchResults.empty("No course found matching 'Invalid Course'")
        mock_vector_store.search.return_value = error_results

        # Execute search
        result = course_search_tool.execute(
            query="anything",
            course_name="Invalid Course"
        )

        # Should return error message
        assert "No course found" in result or result.startswith("No course")

    def test_execute_vector_store_error(self, course_search_tool, mock_vector_store):
        """Test handling of vector store errors"""
        # Configure mock to return results with error
        error_results = SearchResults.empty("Search error: Database connection failed")
        mock_vector_store.search.return_value = error_results

        # Execute search
        result = course_search_tool.execute(query="test")

        # Should return error message
        assert "error" in result.lower() or "Search error" in result

    def test_source_tracking(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test that sources are tracked correctly"""
        # Execute search
        result = course_search_tool.execute(query="machine learning")

        # Verify sources were tracked
        assert len(course_search_tool.last_sources) > 0

        # Verify source structure
        source = course_search_tool.last_sources[0]
        assert "text" in source
        assert "link" in source
        assert "Introduction to Machine Learning" in source["text"]

    def test_result_formatting(self, course_search_tool, mock_vector_store, sample_search_results):
        """Test that results are formatted correctly"""
        # Execute search
        result = course_search_tool.execute(query="machine learning")

        # Verify formatting structure
        # Should have course title in brackets
        assert "[Introduction to Machine Learning" in result

        # Should have lesson information
        assert "Lesson 0" in result or "Lesson 1" in result

        # Should have actual content
        assert "Machine learning" in result

        # Should separate multiple results
        if len(sample_search_results.documents) > 1:
            # Check for double newline separator
            assert "\n\n" in result


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool tool definition"""

    def test_tool_definition_structure(self, course_search_tool):
        """Test that tool definition has correct structure"""
        definition = course_search_tool.get_tool_definition()

        # Verify required fields
        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition

        # Verify tool name
        assert definition["name"] == "search_course_content"

        # Verify input schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_tool_definition_parameters(self, course_search_tool):
        """Test that tool definition has correct parameters"""
        definition = course_search_tool.get_tool_definition()
        properties = definition["input_schema"]["properties"]

        # Verify query parameter
        assert "query" in properties
        assert properties["query"]["type"] == "string"

        # Verify optional parameters
        assert "course_name" in properties
        assert "lesson_number" in properties

        # Verify required fields
        required = definition["input_schema"]["required"]
        assert "query" in required
        assert "course_name" not in required  # Optional
        assert "lesson_number" not in required  # Optional


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        # Register tool
        manager.register_tool(tool)

        # Verify tool is registered
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()

        # Should return list of definitions
        assert isinstance(definitions, list)
        assert len(definitions) >= 1

        # Each definition should have required fields
        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition

    def test_execute_tool(self, tool_manager, mock_vector_store):
        """Test tool execution through manager"""
        # Execute tool via manager
        result = tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )

        # Should return string result
        assert isinstance(result, str)

        # Vector store should have been called
        mock_vector_store.search.assert_called()

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        # Should return error message
        assert "not found" in result.lower()

    def test_get_last_sources(self, tool_manager, mock_vector_store):
        """Test retrieving sources from last search"""
        # Execute a search
        tool_manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = tool_manager.get_last_sources()

        # Should return list of sources
        assert isinstance(sources, list)

    def test_reset_sources(self, tool_manager):
        """Test resetting sources"""
        # Execute a search to populate sources
        tool_manager.execute_tool("search_course_content", query="test")

        # Reset sources
        tool_manager.reset_sources()

        # Sources should be empty
        sources = tool_manager.get_last_sources()
        assert len(sources) == 0
