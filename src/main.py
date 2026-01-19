"""Main entry point for the application."""

from youtube_client import YoutubeClient
from chunk import chunk_transcript_two_tier
from summariser import Summariser
from embedder import Embedder
from vector_store import VectorStore
from rag_chat import RAGChat
import sys
import json
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    # Handle short URLs or other formats
    return url.split("/")[-1].split("?")[0]


def index_video(url: str, verbose: bool = False) -> None:
    """Index a video: fetch transcript, chunk, embed, and store."""
    print("Welcome to Pod2Chat!")
    print(f"Indexing video: {url}")

    # 1. Get the video id from the url
    video_id = extract_video_id(url)

    # 2. Get the transcript from the video
    if verbose:
        print("  Step 1: Fetching transcript from YouTube...")
    client = YoutubeClient()

    try:
        transcript = client.get_transcript(video_id)
        if verbose:
            print(f"  ✓ Transcript fetched: {len(transcript)} segments")
    except Exception as e:
        print("Could not fetch transcript for your video. Please check the video id and try again.")
        return

    # 3. parse transcript
    if verbose:
        print("  Step 2: Parsing transcript segments...")
    jsonl_text = ""
    raw_segments = []
    for entry in transcript:
        entry_dict = {
            "start": entry.start,
            "text": entry.text,
            "duration": entry.duration,
            "end": entry.start + entry.duration,
        }
        jsonl_text += json.dumps(entry_dict) + "\n"
        raw_segments.append(entry_dict)
    if verbose:
        print(f"  ✓ Parsed {len(raw_segments)} segments")

    # 4. Create output folder and write transcript
    if verbose:
        print("  Step 3: Creating output directory...")
    output_folder = f"output_{video_id}"
    os.makedirs(output_folder, exist_ok=True)
    
    with open(f"{output_folder}/transcript.jsonl", "w") as f:
        f.write(jsonl_text)

    print(f"Transcript saved to {output_folder}/transcript.jsonl")

    # 5. chunk transcript
    if verbose:
        print("  Step 4: Chunking transcript into fine and coarse tiers...")
    print("Chunking transcript...")
    chunks = chunk_transcript_two_tier(raw_segments)
    if verbose:
        fine_count = len(chunks.get("fine", []))
        coarse_count = len(chunks.get("coarse", []))
        print(f"  ✓ Created {fine_count} fine chunks and {coarse_count} coarse chunks")
    print(f"Chunks saved to {output_folder}/chunks.jsonl")
    with open(f"{output_folder}/chunks.jsonl", "w") as f:
        json.dump(chunks, f)
    
    # 6. Generate embeddings and store in vector store
    if verbose:
        print("  Step 5: Generating embeddings for chunks...")
    print("Generating embeddings and storing in vector store...")
    try:
        # Initialize embedder
        embedder = Embedder()
        
        # Combine fine and coarse chunks for embedding
        all_chunks = []
        for tier in ["fine", "coarse"]:
            tier_chunks = chunks.get(tier, [])
            for chunk in tier_chunks:
                chunk["tier"] = tier
            all_chunks.extend(tier_chunks)
        
        # Generate embeddings for all chunks
        chunks_with_embeddings = embedder.embed_chunks(all_chunks)
        if verbose:
            print(f"  ✓ Generated embeddings for {len(chunks_with_embeddings)} chunks")
        
        # Store in vector store (with URL)
        db_path = f"{output_folder}/chunks.db"
        if verbose:
            print("  Step 6: Storing chunks in vector database...")
        with VectorStore(db_path) as store:
            store.insert_chunks(chunks_with_embeddings, video_id, url)
        if verbose:
            print(f"  ✓ Vector store created with {len(chunks_with_embeddings)} chunks")
        
        print(f"Vector store saved to {db_path}")
    except Exception as e:
        print(f"Warning: Embedding/vector store generation failed: {e}")
        print("Continuing without embeddings...")

    # 7. Generate markdown summary
    if verbose:
        print("  Step 7: Generating markdown summary...")
    print("Generating markdown summary...")
    try:
        # Get video metadata
        if verbose:
            print("    Fetching video metadata...")
        metadata = client.get_video_metadata(video_id, url)
        if verbose:
            print(f"    Title: {metadata.get('title', 'N/A')}")
            print(f"    Channel: {metadata.get('channel', 'N/A')}")
        
        # Initialize summarizer
        summariser = Summariser(verbose=verbose)
        
        # Generate summary
        summary_markdown = summariser.generate_summary(chunks, metadata, raw_segments)
        
        # Write summary to file
        summary_path = f"{output_folder}/summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_markdown)
        
        print(f"Summary saved to {summary_path}")
    except Exception as e:
        print(f"Warning: Summary generation failed: {e}")
        print("Continuing without summary...")
    
    print(f"\nVideo indexed successfully! You can now chat about it using: pod2chat chat {url}")


def chat_video(url: str, verbose: bool = False) -> None:
    """Start interactive chat about a video."""
    if verbose:
        print("Initializing chat session...")
    video_id = extract_video_id(url)
    output_folder = f"output_{video_id}"
    db_path = f"{output_folder}/chunks.db"
    
    # Check if video is indexed
    if not os.path.exists(db_path):
        print(f"Video not indexed. Index now? (Y/n): ", end="")
        response = input().strip().lower()
        if response in ["", "y", "yes"]:
            print("\nIndexing video first...")
            index_video(url, verbose=verbose)
            print("\nStarting chat...\n")
        else:
            print("Exiting. Please index the video first using: pod2chat index <URL>")
            return
    
    # Initialize RAG chat
    try:
        if verbose:
            print(f"  Loading vector store from {db_path}...")
        rag_chat = RAGChat(db_path, url, verbose=verbose)
        if verbose:
            print("  ✓ RAG chat initialized")
    except Exception as e:
        print(f"Error initializing chat: {e}")
        return
    
    print(f"Chatting about video: {url}")
    print("Type your questions (or '/exit' to quit, '/help' for help)\n")
    
    result = None  # Initialize to track last result
    try:
        while True:
            # Get user input
            query = input("You: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ["/exit", "/quit"]:
                if result:
                    print(f"\nToken usage: {result['input_tokens']:,} input, {result['output_tokens']:,} output")
                    print(f"Cost: ${result['total_cost']:.6f} (input: ${result['input_cost']:.6f}, output: ${result['output_cost']:.6f})")
                print("Goodbye!")
                break
            elif query.lower() == "/help":
                print("Commands:")
                print("  /exit or /quit - Exit chat")
                print("  /help - Show this help")
                continue
            
            # Process query
            try:
                if verbose:
                    print(f"\n  Processing query: '{query}'...")
                result = rag_chat.chat(query)
                
                # Display response
                print(f"\nAssistant: {result['response']}")
                
                # Display sources
                if result['sources']:
                    sources_str = ", ".join(f"[{ts}]" for ts in result['sources'])
                    print(f"\nSources: {sources_str}")
                
                # Display token usage and cost
                if verbose:
                    print(f"  Token usage: {result['input_tokens']:,} input, {result['output_tokens']:,} output")
                    print(f"  Cost: ${result['total_cost']:.6f}")
                print()
                
            except Exception as e:
                
                print(f"Error: {e}\n")
                
    except KeyboardInterrupt:
        if result:
            print(f"\nToken usage: {result['input_tokens']:,} input, {result['output_tokens']:,} output")
            print(f"Cost: ${result['total_cost']:.6f} (input: ${result['input_cost']:.6f}, output: ${result['output_cost']:.6f})")
        print("\n\nGoodbye!")
    finally:
        rag_chat.close()


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Pod2Chat - Index and chat about YouTube videos",
        prog="pod2chat"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a YouTube video")
    index_parser.add_argument("url", help="YouTube video URL")
    index_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat about a YouTube video")
    chat_parser.add_argument("url", help="YouTube video URL")
    chat_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.command == "index":
        index_video(args.url, verbose=args.verbose)
    elif args.command == "chat":
        chat_video(args.url, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
