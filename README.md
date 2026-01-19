# Pod2Chat

A Python application for indexing YouTube videos and chatting about them using RAG (Retrieval-Augmented Generation).

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your API keys (e.g., OpenAI API key for embeddings and chat functionality).

## Usage

The application is accessed through `src/main.py` with two main commands:

### Index a YouTube Video

Index a YouTube video by fetching its transcript, chunking it, generating embeddings, and creating a vector store:

```bash
python -m src.main index <YOUTUBE_URL>
```

Example:
```bash
python -m src.main index https://www.youtube.com/watch?v=VIDEO_ID
```

This will:
- Fetch the video transcript
- Chunk the transcript into fine and coarse segments
- Generate embeddings for all chunks
- Store chunks in a vector database
- Generate a markdown summary

Output files are saved in an `output_<VIDEO_ID>/` directory.

### Chat About a Video

Start an interactive chat session about an indexed video:

```bash
python -m src.main chat <YOUTUBE_URL>
```

Example:
```bash
python -m src.main chat https://www.youtube.com/watch?v=VIDEO_ID
```

If the video hasn't been indexed yet, you'll be prompted to index it first.

During the chat session:
- Type your questions to ask about the video content
- Use `/help` to see available commands
- Use `/exit` or `/quit` to exit the chat

## Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest
```

## Project Structure

```
pod2chat/
├── src/              # Source code
│   ├── __init__.py
│   ├── main.py       # Main entry point (CLI interface)
│   ├── youtube_client.py
│   ├── chunk.py
│   ├── summarizer.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── rag_chat.py
├── tests/            # Test files
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt  # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py          # Package setup configuration
├── .gitignore        # Git ignore rules
└── README.md         # This file
```
TODO: 
 - Fix summariser deep dive
    1. Get topics - sliding window technique where a topic for the first window, slide through chunks until topic change. Mark chunks under a respective topic 
    2. Build ex synopsis
    3. For each topic, extract actionable insights that should be applied in a professional capacity. THings that should be remembered that would make the reader more productive, or have a higher quality output??
    4. 