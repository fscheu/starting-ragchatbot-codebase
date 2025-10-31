# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) chatbot system that enables semantic search across educational course materials using Claude AI and ChromaDB vector storage. The system uses a tool-based retrieval architecture where Claude actively decides when to search for information rather than always retrieving context upfront.

## Development Setup

### Prerequisites
- Python 3.13+
- uv (Python package manager)
- Anthropic API key

### Installation
```cmd
# Install dependencies
uv sync

# Set up environment (copy example file)
copy .env.example .env

# Edit .env and add your ANTHROPIC_API_KEY
notepad .env
```

### Running the Application
```cmd
# Quick start (Git Bash)
bash run.sh

# Manual start (Command Prompt/PowerShell - from root)
cd backend
uv run uvicorn app:app --reload --port 8000

# Access points
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

## Architecture

### RAG Pipeline Flow

1. **Document Ingestion** (startup): `docs/` folder → DocumentProcessor → VectorStore (ChromaDB)
2. **Query Processing**: User query → RAGSystem → AIGenerator (Claude with tools)
3. **Tool-Based Retrieval**: Claude calls `search_course_content` tool → VectorStore semantic search
4. **Response Generation**: Claude synthesizes answer from retrieved context
5. **Session Management**: Last 2 Q&A exchanges maintained per session

### Core Components

**RAGSystem** (rag_system.py) - Main orchestrator
- Coordinates all components
- Entry point: `query(query, session_id)` returns `(answer, sources)`
- Initializes tool-based search architecture

**AIGenerator** (ai_generator.py) - Claude integration
- Handles tool-calling loop with Claude API
- Model: claude-sonnet-4-20250514 (configurable in config.py)
- System prompt emphasizes: one search per query, concise responses, no meta-commentary
- Tool execution flow: initial API call → tool_use → execute tools → final API call with results

**VectorStore** (vector_store.py) - ChromaDB wrapper with two collections
- `course_catalog`: Course metadata (title, instructor, lessons) - used for fuzzy course name matching
- `course_content`: Chunked content (800 chars, 100 overlap) - used for semantic search
- Main interface: `search(query, course_name=None, lesson_number=None)` returns SearchResults
- Course name resolution: Uses semantic search on catalog to handle partial/fuzzy matches

**DocumentProcessor** (document_processor.py) - Document parsing
- Expected format: Course Title / Course Link / Course Instructor / Lesson markers
- Sentence-based chunking with overlap (preserves semantic boundaries)
- Adds context to chunks: "Course [title] Lesson [num] content: [text]"

**Search Tools** (search_tools.py) - Tool interface
- `CourseSearchTool`: Implements Claude tool interface for course search
- `ToolManager`: Registers tools and handles execution
- Tool tracks sources for frontend display

**SessionManager** (session_manager.py) - Conversation state
- In-memory session storage (cleared on restart)
- Maintains last 2 exchanges (4 messages total) per session
- Simple counter-based session IDs

### Data Models (models.py)

```python
Course(title, course_link, instructor, lessons: List[Lesson])
Lesson(lesson_number, title, lesson_link)
CourseChunk(content, course_title, lesson_number, chunk_index)
```

## Key Design Patterns

### Two-Collection Strategy
The vector store separates course metadata (catalog) from content. This enables:
1. Fuzzy course name matching via semantic search on titles
2. Efficient filtering of content search by resolved course title
3. Clean separation between "what courses exist" vs "what's in courses"

### Tool-Based vs Traditional RAG
Unlike traditional RAG (always retrieve → generate), this uses Claude's tool-calling:
- Claude decides IF and WHEN to search based on query type
- General knowledge questions answered directly without search
- Course-specific questions trigger tool use
- More efficient: no unnecessary retrievals

### Chunk Context Enhancement
Each chunk includes metadata in its content:
- First chunk of lesson: "Lesson [num] content: [text]"
- Other chunks: "Course [title] Lesson [num] content: [text]"
- Improves retrieval relevance and response quality

## Configuration (backend/config.py)

All settings centralized in Config dataclass:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (SentenceTransformers)
- `CHUNK_SIZE`: 800 chars, `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges
- `CHROMA_PATH`: "./chroma_db"

## API Endpoints (backend/app.py)

```
POST /api/query
  Body: { query: str, session_id?: str }
  Returns: { answer: str, sources: List[str], session_id: str }

GET /api/courses
  Returns: { total_courses: int, course_titles: List[str] }
```

## Document Format

Course documents in `docs/` should follow:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[content]

Lesson 1: [title]
...
```

Parser is flexible: handles missing metadata, uses filename as fallback title.

## Adding New Features

### Adding New Tools
1. Create class inheriting from `Tool` in search_tools.py
2. Implement `get_tool_definition()` and `execute(**kwargs)`
3. Register in RAGSystem.__init__: `self.tool_manager.register_tool(YourTool())`

### Modifying Search Behavior
- Search logic: `VectorStore.search()` and `VectorStore._resolve_course_name()`
- Tool parameters: `CourseSearchTool.get_tool_definition()`
- Result formatting: `CourseSearchTool._format_results()`

### Changing AI Behavior
- System prompt: `AIGenerator.SYSTEM_PROMPT` (static class variable)
- Temperature/tokens: `AIGenerator.base_params`
- Tool calling logic: `AIGenerator._handle_tool_execution()`

## Common Gotchas

1. **ChromaDB persistence**: Data persists in `./chroma_db`. Delete this folder to reset the database.
2. **Duplicate courses**: System checks existing titles before adding. Documents auto-load on startup (no duplicate check across restarts).
3. **Session lifetime**: Sessions only exist in memory. Restart clears all sessions.
4. **Course name matching**: Uses fuzzy semantic matching. "Introduction" may match "Introduction to AI Course".
5. **Shell scripts**: The `run.sh` script requires Git Bash on Windows. Alternatively, use the manual start method with Command Prompt or PowerShell.
6. **Chunk context**: Modifying chunk format requires reindexing all documents (clear DB and restart).
