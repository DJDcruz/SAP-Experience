"""
Test script to query ChromaDB for restaurants in Dubai.
"""
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'knowledge_base'))

from wikivoyage_chromadb_bot import ChromaDBManager
from services.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def main():
    # Initialize ChromaDB directly to access raw search with distances
    print(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    db = ChromaDBManager(db_path=CHROMA_PERSIST_DIR, model_name=EMBEDDING_MODEL)
    
    # Query parameters
    location = "dubai"
    query = "restaurants can i eat at here"
    
    # Test 1: Combined search (current behavior)
    search_text = f"{location} {query}"
    print(f"\n{'='*60}")
    print(f"TEST 1: Combined search")
    print(f"Search text: '{search_text}'")
    print(f"{'='*60}\n")
    
    results = db.search(search_text, n_results=5)
    print_results(results)
    
    # Test 2: Search with metadata filter for title containing "Dubai"
    print(f"\n{'='*60}")
    print(f"TEST 2: Search with title filter (where title contains 'Dubai')")
    print(f"Search text: '{query}'")
    print(f"{'='*60}\n")
    
    # Query with metadata filter
    query_embedding = db.embedder.encode([query], convert_to_numpy=True)
    filtered_results = db.collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        where={"title": {"$eq": "Dubai"}}  # Exact match filter
    )
    print_raw_results(filtered_results)
    
    # Test 3: Check what Dubai documents exist
    print(f"\n{'='*60}")
    print(f"TEST 3: Search just for 'Dubai' to see what exists")
    print(f"{'='*60}\n")
    
    dubai_results = db.search("Dubai", n_results=5)
    print_results(dubai_results)

def print_results(results):
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        title = doc.get('title', 'No title')
        distance = doc.get('distance', 'N/A')
        content = doc.get('content', '')[:300]
        print(f"--- Result {i} ---")
        print(f"Title: {title}")
        print(f"Distance: {distance}")
        print(f"Content: {content}...")
        print()

def print_raw_results(results):
    if not results['documents'] or not results['documents'][0]:
        print("No results found with filter.")
        return
    
    print(f"Found {len(results['documents'][0])} results:\n")
    for i, doc in enumerate(results['documents'][0]):
        title = results['metadatas'][0][i]['title']
        distance = results['distances'][0][i] if results['distances'] else 'N/A'
        print(f"--- Result {i+1} ---")
        print(f"Title: {title}")
        print(f"Distance: {distance}")
        print(f"Content: {doc[:300]}...")
        print()

if __name__ == "__main__":
    main()
