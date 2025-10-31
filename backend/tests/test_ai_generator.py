"""Unit tests for AIGenerator"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Tests for basic AIGenerator functionality"""

    def test_generate_without_tools(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test generating response without tools"""
        # Configure mock for direct response
        mock_direct_response = Mock()
        mock_content = Mock()
        mock_content.text = "Direct answer"
        mock_direct_response.content = [mock_content]
        mock_direct_response.stop_reason = "end_turn"
        # Reset side_effect and set return_value
        mock_anthropic_client.messages.create.side_effect = None
        mock_anthropic_client.messages.create.return_value = mock_direct_response

        # Generate response without tools
        response = ai_generator_with_mock.generate_response(
            query="What is machine learning?", tools=None
        )

        # Should return direct answer
        assert response == "Direct answer"

        # Should have called API once
        assert mock_anthropic_client.messages.create.call_count == 1

        # Verify API call parameters
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "messages" in call_args
        assert call_args["messages"][0]["role"] == "user"
        assert "What is machine learning?" in call_args["messages"][0]["content"]

    def test_system_prompt_inclusion(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test that system prompt is included in API call"""
        # Configure mock
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        ai_generator_with_mock.generate_response(query="Test query")

        # Verify system prompt was included
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "system" in call_args
        assert "AI assistant specialized in course materials" in call_args["system"]

    def test_conversation_history_integration(
        self, ai_generator_with_mock, mock_anthropic_client
    ):
        """Test conversation history is included in system context"""
        # Configure mock
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response with history
        history = "User: Previous question\nAssistant: Previous answer"
        ai_generator_with_mock.generate_response(
            query="Follow-up question", conversation_history=history
        )

        # Verify history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]


class TestAIGeneratorToolCalling:
    """Tests for tool calling functionality"""

    def test_generate_triggers_tool(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that tool use is triggered correctly"""
        # Configure mock for tool use
        mock_tool_response = Mock()
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer")]
        mock_final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Generate response with tools
        tools = tool_manager.get_tool_definitions()
        response = ai_generator_with_mock.generate_response(
            query="What is machine learning in the course?",
            tools=tools,
            tool_manager=tool_manager,
        )

        # Should return final response after tool execution
        assert response == "Final answer"

        # Should have called API twice (initial + final)
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_tool_choice_auto(self, ai_generator_with_mock, mock_anthropic_client):
        """Test that tool_choice is set to auto when tools provided"""
        # Configure mock
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate with tools
        tools = [{"name": "test_tool", "description": "Test"}]
        ai_generator_with_mock.generate_response(query="Test", tools=tools)

        # Verify tool_choice was set
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tool_choice" in call_args
        assert call_args["tool_choice"]["type"] == "auto"

    def test_tool_execution_loop(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test complete tool execution loop"""
        # Configure mock for tool use
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_abc"
        tool_block.input = {"query": "supervised learning"}
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Based on the search...")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final,
        ]

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="Tell me about supervised learning",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify complete loop executed
        assert mock_anthropic_client.messages.create.call_count == 2

        # Verify second call included tool results
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert len(second_call_args["messages"]) == 3  # user + assistant + tool_result

        # Verify tool result message structure
        tool_result_msg = second_call_args["messages"][2]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"

    def test_tool_result_processing(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that tool results are processed correctly"""
        # Configure mocks
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_xyz"
        tool_block.input = {"query": "test"}
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Answer with tool results")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final,
        ]

        # Execute
        response = ai_generator_with_mock.generate_response(
            query="Query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Verify tool result was added to messages
        second_call = mock_anthropic_client.messages.create.call_args_list[1][1]
        tool_result = second_call["messages"][2]["content"][0]

        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_xyz"
        assert "content" in tool_result


class TestAIGeneratorErrorHandling:
    """Tests for error handling"""

    def test_api_error_handling(self, ai_generator_with_mock, mock_anthropic_client):
        """Test handling of Anthropic API errors"""
        # Configure mock to raise error
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        # Should raise exception
        with pytest.raises(Exception) as exc_info:
            ai_generator_with_mock.generate_response(query="Test")

        assert "API Error" in str(exc_info.value)

    def test_tool_execution_error(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test handling of tool execution errors"""
        # Configure mock for tool use
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Final answer")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final,
        ]

        # Mock tool to raise error
        with patch.object(
            tool_manager, "execute_tool", side_effect=Exception("Tool failed")
        ):
            # Error should be handled gracefully (not raised)
            response = ai_generator_with_mock.generate_response(
                query="Test",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            # Should still return a response
            assert response == "Final answer"

            # Verify error was passed to Claude as tool result
            second_call = mock_anthropic_client.messages.create.call_args_list[1][1]
            tool_results = second_call["messages"][2]["content"]
            assert "is_error" in tool_results[0]
            assert tool_results[0]["is_error"] is True
            assert "Tool failed" in tool_results[0]["content"]

    def test_malformed_tool_response(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test handling of malformed tool responses"""
        # Configure mock with malformed tool response
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = None  # Missing ID
        tool_block.input = {}  # Empty input
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_anthropic_client.messages.create.return_value = mock_tool_response

        # Should handle gracefully or raise appropriate error
        # This tests current behavior (may expose bugs)
        try:
            response = ai_generator_with_mock.generate_response(
                query="Test",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )
            # If it succeeds, verify it handled the issue
            assert isinstance(response, str)
        except (AttributeError, KeyError, TypeError) as e:
            # If it fails, this indicates missing validation
            pytest.fail(f"Should handle malformed responses gracefully: {e}")


class TestAIGeneratorParameters:
    """Tests for API parameters"""

    def test_model_parameter(self, test_config, mock_anthropic_client):
        """Test that correct model is used"""
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_anthropic_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        generator.generate_response(query="Test")

        # Verify model parameter
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["model"] == test_config.ANTHROPIC_MODEL

    def test_temperature_parameter(self, ai_generator_with_mock, mock_anthropic_client):
        """Test that temperature is set correctly"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        ai_generator_with_mock.generate_response(query="Test")

        # Verify temperature
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "temperature" in call_args
        assert call_args["temperature"] == 0

    def test_max_tokens_parameter(self, ai_generator_with_mock, mock_anthropic_client):
        """Test that max_tokens is set correctly"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        ai_generator_with_mock.generate_response(query="Test")

        # Verify max_tokens
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "max_tokens" in call_args
        assert call_args["max_tokens"] == 800


class TestAIGeneratorSequentialToolCalling:
    """Tests for sequential tool calling functionality"""

    def test_sequential_two_tool_calls(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that Claude can make 2 sequential tool calls"""
        # Mock response 1: Claude requests first tool
        mock_tool_response_1 = Mock()
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "get_course_outline"
        tool_block_1.id = "tool_001"
        tool_block_1.input = {"course_name": "Machine Learning"}
        mock_tool_response_1.content = [tool_block_1]
        mock_tool_response_1.stop_reason = "tool_use"

        # Mock response 2: After first tool, Claude requests second tool
        mock_tool_response_2 = Mock()
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_002"
        tool_block_2.input = {"query": "supervised learning"}
        mock_tool_response_2.content = [tool_block_2]
        mock_tool_response_2.stop_reason = "tool_use"

        # Mock response 3: Final text response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final answer after two tool calls")]
        mock_final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final_response,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        response = ai_generator_with_mock.generate_response(
            query="Search for a course that discusses the same topic as lesson 4 of Machine Learning",
            tools=tools,
            tool_manager=tool_manager,
        )

        # Verify response
        assert response == "Final answer after two tool calls"

        # Verify 3 API calls were made (initial + round 1 + round 2)
        assert mock_anthropic_client.messages.create.call_count == 3

        # Verify tools parameter was included in first two calls
        first_call = mock_anthropic_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call
        assert first_call["tool_choice"]["type"] == "auto"

        second_call = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call
        assert second_call["tool_choice"]["type"] == "auto"

    def test_max_rounds_enforced(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that system stops after 2 tool rounds even if Claude wants more"""
        # Mock: Each response keeps requesting tools
        mock_tool_response_1 = Mock()
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_001"
        tool_block_1.input = {"query": "test1"}
        mock_tool_response_1.content = [tool_block_1]
        mock_tool_response_1.stop_reason = "tool_use"

        mock_tool_response_2 = Mock()
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_002"
        tool_block_2.input = {"query": "test2"}
        mock_tool_response_2.content = [tool_block_2]
        mock_tool_response_2.stop_reason = "tool_use"

        # After 2 rounds, final response (no more tool_use)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Max rounds reached")]
        mock_final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final_response,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        response = ai_generator_with_mock.generate_response(
            query="Test query", tools=tools, tool_manager=tool_manager
        )

        # Verify exactly 3 API calls (initial + 2 rounds max)
        assert mock_anthropic_client.messages.create.call_count == 3

        # Verify third call does NOT include tools (max rounds reached)
        third_call = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call

    def test_early_termination_after_one_tool(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that system stops when Claude returns text instead of tool_use"""
        # Mock response 1: Tool use
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_001"
        tool_block.input = {"query": "test"}
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        # Mock response 2: Text response (no more tools)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Answer after one tool")]
        mock_final_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        response = ai_generator_with_mock.generate_response(
            query="Simple query", tools=tools, tool_manager=tool_manager
        )

        # Verify only 2 API calls (initial + 1 round)
        assert mock_anthropic_client.messages.create.call_count == 2
        assert response == "Answer after one tool"

    def test_tools_parameter_preserved_in_rounds(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that tools parameter is correctly included in intermediate rounds"""
        # Mock tool responses
        mock_tool_response_1 = Mock()
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_001"
        tool_block_1.input = {"query": "test1"}
        mock_tool_response_1.content = [tool_block_1]
        mock_tool_response_1.stop_reason = "tool_use"

        mock_tool_response_2 = Mock()
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_002"
        tool_block_2.input = {"query": "test2"}
        mock_tool_response_2.content = [tool_block_2]
        mock_tool_response_2.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Done")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        ai_generator_with_mock.generate_response(
            query="Test", tools=tools, tool_manager=tool_manager
        )

        # Verify tools in first round
        first_round_call = mock_anthropic_client.messages.create.call_args_list[0][1]
        assert "tools" in first_round_call
        assert first_round_call["tools"] == tools

        # Verify tools in second round
        second_round_call = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_round_call
        assert second_round_call["tools"] == tools

    def test_context_accumulation_across_rounds(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test that conversation context grows correctly across rounds"""
        # Setup mocks
        mock_tool_response_1 = Mock()
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_001"
        tool_block_1.input = {"query": "test1"}
        mock_tool_response_1.content = [tool_block_1]
        mock_tool_response_1.stop_reason = "tool_use"

        mock_tool_response_2 = Mock()
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_002"
        tool_block_2.input = {"query": "test2"}
        mock_tool_response_2.content = [tool_block_2]
        mock_tool_response_2.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Done")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response_1,
            mock_tool_response_2,
            mock_final,
        ]

        # Execute
        tools = tool_manager.get_tool_definitions()
        ai_generator_with_mock.generate_response(
            query="Test query", tools=tools, tool_manager=tool_manager
        )

        # Check message structure in second round call
        second_call_messages = mock_anthropic_client.messages.create.call_args_list[1][
            1
        ]["messages"]

        # Should have: initial user message, assistant tool_use, user tool_result
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"  # Initial query
        assert second_call_messages[1]["role"] == "assistant"  # Tool use 1
        assert second_call_messages[2]["role"] == "user"  # Tool result 1

        # Check message structure in third round call
        third_call_messages = mock_anthropic_client.messages.create.call_args_list[2][
            1
        ]["messages"]

        # Should have: initial user, assistant tool_use 1, user tool_result 1, assistant tool_use 2, user tool_result 2
        assert len(third_call_messages) == 5
        assert third_call_messages[0]["role"] == "user"  # Initial query
        assert third_call_messages[1]["role"] == "assistant"  # Tool use 1
        assert third_call_messages[2]["role"] == "user"  # Tool result 1
        assert third_call_messages[3]["role"] == "assistant"  # Tool use 2
        assert third_call_messages[4]["role"] == "user"  # Tool result 2

    def test_tool_execution_error_in_sequential_calls(
        self, ai_generator_with_mock, mock_anthropic_client, tool_manager
    ):
        """Test graceful handling of tool execution errors in sequential calls"""
        # Mock tool response that will cause error
        mock_tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_001"
        tool_block.input = {"query": "test"}
        mock_tool_response.content = [tool_block]
        mock_tool_response.stop_reason = "tool_use"

        mock_final = Mock()
        mock_final.content = [Mock(text="Error handled")]
        mock_final.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final,
        ]

        # Mock tool to raise error
        with patch.object(
            tool_manager, "execute_tool", side_effect=Exception("Tool error")
        ):
            # Execute
            tools = tool_manager.get_tool_definitions()
            response = ai_generator_with_mock.generate_response(
                query="Test", tools=tools, tool_manager=tool_manager
            )

            # Should still return response
            assert response == "Error handled"

            # Verify error was included in tool result
            second_call = mock_anthropic_client.messages.create.call_args_list[1][1]
            tool_results = second_call["messages"][2]["content"]
            assert "is_error" in tool_results[0]
            assert tool_results[0]["is_error"] is True
            assert "Tool error" in tool_results[0]["content"]
