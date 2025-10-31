"""Integration tests for RAG system"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from config import Config


class TestRAGSystemContentQueries:
    """Tests for content-related query handling"""

    def test_content_query_flow(self, test_config):
        """Test end-to-end content query flow"""
        # Create RAG system with mocked components
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            # Configure mocks
            mock_store = MockVectorStore.return_value
            mock_store.search.return_value = Mock(
                documents=["Machine learning content"],
                metadata=[{"course_title": "ML Course", "lesson_number": 0}],
                distances=[0.1],
                error=None,
                is_empty=Mock(return_value=False)
            )
            mock_store._resolve_course_name.return_value = "ML Course"
            mock_store.get_lesson_link.return_value = "https://example.com/lesson-0"

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.return_value = "Machine learning is a subset of AI..."

            # Create system and execute query
            rag = RAGSystem(test_config)
            answer, sources = rag.query("What is machine learning?", session_id="test_1")

            # Verify response
            assert isinstance(answer, str)
            assert len(answer) > 0

            # Verify AI generator was called with tools
            mock_ai.generate_response.assert_called_once()
            call_kwargs = mock_ai.generate_response.call_args[1]
            assert 'tools' in call_kwargs
            assert 'tool_manager' in call_kwargs

    def test_content_query_with_sources(self, test_config):
        """Test that sources are returned correctly"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            # Configure mocks to simulate tool use
            mock_store = MockVectorStore.return_value
            mock_ai = MockAIGenerator.return_value

            # Simulate tool execution by directly calling search tool
            rag = RAGSystem(test_config)

            # Mock the search tool to populate sources
            rag.search_tool.last_sources = [
                {"text": "ML Course - Lesson 0", "link": "https://example.com/lesson-0"}
            ]

            # Mock AI response
            mock_ai.generate_response.return_value = "Answer based on course materials"

            # Execute query
            answer, sources = rag.query("Test query", session_id="test_2")

            # Verify sources were returned
            assert isinstance(sources, list)

    def test_empty_results_handling(self, test_config):
        """Test handling when search returns no results"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_store = MockVectorStore.return_value
            mock_ai = MockAIGenerator.return_value

            # Configure AI to handle empty search
            mock_ai.generate_response.return_value = "No relevant content found in the courses."

            rag = RAGSystem(test_config)
            answer, sources = rag.query("Nonexistent topic", session_id="test_3")

            # Should still return an answer
            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_error_propagation(self, test_config):
        """Test error propagation through the stack"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAIGenerator.return_value
            mock_ai.generate_response.side_effect = Exception("API Error")

            rag = RAGSystem(test_config)

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                rag.query("Test query", session_id="test_4")

            assert "API Error" in str(exc_info.value)


class TestRAGSystemSessionManagement:
    """Tests for session and conversation management"""

    def test_session_management(self, test_config):
        """Test session creation and management"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Answer"

            rag = RAGSystem(test_config)

            # Query with specific session
            session_id = "session_123"
            answer1, _ = rag.query("First question", session_id=session_id)

            # Verify session was used
            assert mock_ai.generate_response.call_count == 1

            # Second query with same session
            answer2, _ = rag.query("Second question", session_id=session_id)

            # Verify history was passed on second call
            second_call_kwargs = mock_ai.generate_response.call_args[1]
            assert 'conversation_history' in second_call_kwargs

    def test_conversation_context(self, test_config):
        """Test multi-turn conversation with context"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Answer"

            rag = RAGSystem(test_config)
            session_id = "conv_session"

            # First exchange
            rag.query("What is supervised learning?", session_id=session_id)

            # Second exchange should include history
            rag.query("Give me an example", session_id=session_id)

            # Verify second call included conversation history
            second_call = mock_ai.generate_response.call_args_list[1]
            call_kwargs = second_call[1]

            assert call_kwargs['conversation_history'] is not None
            history = call_kwargs['conversation_history']
            assert "What is supervised learning?" in history

    def test_concurrent_sessions(self, test_config):
        """Test handling multiple concurrent sessions"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Answer"

            rag = RAGSystem(test_config)

            # Create multiple sessions
            rag.query("Question 1", session_id="session_A")
            rag.query("Question 2", session_id="session_B")
            rag.query("Follow-up A", session_id="session_A")

            # Verify sessions are independent
            # Session A should have history, session B should be fresh
            assert rag.session_manager.sessions["session_A"] is not None
            assert rag.session_manager.sessions["session_B"] is not None
            assert len(rag.session_manager.sessions["session_A"]) > len(rag.session_manager.sessions["session_B"])


class TestRAGSystemToolIntegration:
    """Tests for tool integration in RAG system"""

    def test_tool_manager_integration(self, test_config):
        """Test that tools are properly registered"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):

            rag = RAGSystem(test_config)

            # Verify tools are registered
            definitions = rag.tool_manager.get_tool_definitions()
            assert len(definitions) >= 1

            # Verify search tool is registered
            tool_names = [d["name"] for d in definitions]
            assert "search_course_content" in tool_names

    def test_outline_tool_integration(self, test_config):
        """Test course outline tool integration"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):

            mock_store = MockVectorStore.return_value

            rag = RAGSystem(test_config)

            # Verify outline tool is registered
            definitions = rag.tool_manager.get_tool_definitions()
            tool_names = [d["name"] for d in definitions]
            assert "get_course_outline" in tool_names

    def test_tool_execution_through_rag(self, test_config):
        """Test tool execution through RAG system"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_store = MockVectorStore.return_value
            mock_ai = MockAI.return_value

            # Configure mock to simulate tool calling
            def mock_generate_with_tool(*args, **kwargs):
                # Simulate tool execution
                if 'tool_manager' in kwargs:
                    tool_mgr = kwargs['tool_manager']
                    # Execute search tool
                    tool_mgr.execute_tool("search_course_content", query="test")
                return "Answer using tool results"

            mock_ai.generate_response.side_effect = mock_generate_with_tool

            rag = RAGSystem(test_config)
            answer, sources = rag.query("Search query", session_id="tool_test")

            # Verify tool was available
            assert mock_ai.generate_response.call_count == 1


class TestRAGSystemQueryTypes:
    """Tests for different query types"""

    def test_general_knowledge_query(self, test_config):
        """Test query that doesn't require course search"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "General knowledge answer without using tools"

            rag = RAGSystem(test_config)
            answer, sources = rag.query("What is 2+2?", session_id="general_test")

            # Should return answer
            assert isinstance(answer, str)
            # Sources may be empty if no tool used
            assert isinstance(sources, list)

    def test_course_specific_query(self, test_config):
        """Test query requiring course content"""
        with patch('rag_system.VectorStore') as MockStore, \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_store = MockStore.return_value
            mock_ai = MockAI.return_value

            # Simulate tool use by AI
            def mock_tool_usage(*args, **kwargs):
                if 'tool_manager' in kwargs:
                    # Populate sources through tool
                    kwargs['tool_manager'].tools['search_course_content'].last_sources = [
                        {"text": "Source", "link": "http://link"}
                    ]
                return "Course-specific answer"

            mock_ai.generate_response.side_effect = mock_tool_usage

            rag = RAGSystem(test_config)
            answer, sources = rag.query(
                "What is supervised learning in the ML course?",
                session_id="course_test"
            )

            # Should have used tool and returned sources
            assert isinstance(answer, str)

    def test_outline_query(self, test_config):
        """Test course outline query"""
        with patch('rag_system.VectorStore') as MockStore, \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_store = MockStore.return_value
            mock_ai = MockAI.return_value

            # Configure outline tool response
            mock_store._resolve_course_name.return_value = "ML Course"
            mock_store.course_catalog = Mock()
            mock_store.course_catalog.get.return_value = {
                'metadatas': [{
                    'title': 'ML Course',
                    'course_link': 'http://link',
                    'instructor': 'Teacher',
                    'lessons_json': '[{"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "http://lesson"}]'
                }]
            }

            mock_ai.generate_response.return_value = "Course outline response"

            rag = RAGSystem(test_config)
            answer, sources = rag.query(
                "What are the lessons in ML course?",
                session_id="outline_test"
            )

            # Should return answer
            assert isinstance(answer, str)


class TestRAGSystemEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_query_without_session_id(self, test_config):
        """Test query without providing session ID"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Answer"

            rag = RAGSystem(test_config)
            answer, sources = rag.query("Test query", session_id=None)

            # Should still work (no history used)
            assert isinstance(answer, str)

    def test_empty_query(self, test_config):
        """Test handling of empty query"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Please provide a question"

            rag = RAGSystem(test_config)
            answer, sources = rag.query("", session_id="empty_test")

            # Should handle gracefully
            assert isinstance(answer, str)

    def test_very_long_query(self, test_config):
        """Test handling of very long query"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.DocumentProcessor'):

            mock_ai = MockAI.return_value
            mock_ai.generate_response.return_value = "Answer to long query"

            rag = RAGSystem(test_config)
            long_query = "What is machine learning? " * 100  # Very long query

            answer, sources = rag.query(long_query, session_id="long_test")

            # Should handle without error
            assert isinstance(answer, str)
