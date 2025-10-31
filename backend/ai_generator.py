import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information and retrieving course outlines.

Tool Usage Guidelines:
- **Content Search Tool** (`search_course_content`): For questions about specific course content, concepts, or detailed educational materials
- **Course Outline Tool** (`get_course_outline`): For questions about course structure, lesson lists, course overview, or "what lessons are in this course"
  - When using this tool, ALWAYS include in your response: course title, course link, and the complete list of lessons with their numbers and titles
  - Format the outline clearly for easy reading
- **Multi-step queries**: You can make up to 2 tool calls in sequence when needed for complex questions
  - Use first tool call to gather initial information (e.g., get course outline to find lesson title)
  - Use second tool call to make a more targeted search based on first results (e.g., search for that topic in other courses)
  - Example: "Search for a course that discusses the same topic as lesson 4 of course X" → get outline for course X → use lesson 4 title to search other courses
- **Synthesize tool results** into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use outline tool first, then present the full course structure including title, link, and all lessons
- **Course content questions**: Use search tool first, then answer
- **Multi-step questions**: Break down into sequential tool calls as needed (maximum 2 calls)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool usage explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager, tools)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, tools: Optional[List]):
        """
        Handle execution of tool calls with support for sequential rounds (max 2).

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            tools: Available tools for Claude to use

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        rounds_completed = 0
        MAX_ROUNDS = 2

        # Loop for up to MAX_ROUNDS of tool calls
        while rounds_completed < MAX_ROUNDS and current_response.stop_reason == "tool_use":
            # Add AI's tool use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True
                        })

            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            rounds_completed += 1

            # Prepare next API call with tools enabled (allowing another round)
            next_params = {
                **self.base_params,
                "messages": messages.copy(),  # Copy to preserve snapshot for testing/debugging
                "system": base_params["system"]
            }

            # Include tools for potential next round if we haven't hit max
            if rounds_completed < MAX_ROUNDS and tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            # Get next response
            current_response = self.client.messages.create(**next_params)

        # Extract and return final text response
        return current_response.content[0].text