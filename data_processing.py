from tqdm import tqdm
from llama_index.core.schema import TextNode, Document
import time
from llama_index.core import StorageContext
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import VectorStoreIndex, Settings
import os
from dotenv import load_dotenv
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

load_dotenv()

start = time.time()

# Get environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MILVUS_URI = os.getenv('MILVUS_URI')
MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')

# Initialize Gemini LLM and Embedding models
llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY)
embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=GEMINI_API_KEY)

# Set the Gemini LLM as the default LLM in Settings
Settings.llm = llm
Settings.embed_model = embed_model

# Initialize Milvus vector store
vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN, 
    overwrite=True,  # Set to True to overwrite existing collection
    collection_name=os.getenv('COLLECTION'),
    dim=768
)

# Function to read files from directory
def read_files_from_directory(directory_path):
    documents = []
    # Make sure directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
        
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    if not files:
        raise ValueError(f"No .txt files found in {directory_path}")
        
    for filename in tqdm(files, desc="Reading files"):
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()  # Remove extra whitespace
                if not content:  # Check if content is empty
                    print(f"Warning: Empty file {filename}")
                    continue
                    
                video_id = os.path.splitext(filename)[0]
                documents.append(Document(text=content, metadata={"source": video_id}))
                print(f"Added document for {video_id} with content length: {len(content)}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            
    return documents

# Main processing
print("Reading documents...")
data_dir = os.getenv('OUTPUT_URL')
documents = read_files_from_directory(data_dir)

print(f"Number of documents read: {len(documents)}")

# Verify documents have content
for doc in documents[:3]:  # Print first 3 docs as sample
    print(f"Document {doc.metadata['source']}: {doc.text[:100]}...")

# Create nodes
print("Creating nodes...")
nodes = []
for doc in tqdm(documents, desc="Creating nodes"):
    try:
        node = TextNode(text=doc.text, metadata=doc.metadata)
        nodes.append(node)
    except Exception as e:
        print(f"Error creating node for {doc.metadata['source']}: {str(e)}")

print(f"Number of nodes created: {len(nodes)}")

# Add error handling and verification
try:
    # After creating nodes, add this:
    print("\nDebug: Checking nodes content...")
    for i, node in enumerate(nodes[:2]):
        print(f"Node {i}: ID={node.node_id}, Text length={len(node.text)}")

    # Create document store explicitly
    doc_store = SimpleDocumentStore()
    for node in nodes:
        doc_store.add_documents([node])

    print(f"\nDebug: Document store has {len(doc_store.docs)} documents")

    # Create storage context with explicit doc store
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store
    )

    # Create index with storage context
    print("Creating index...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )

    # Add verification before saving
    print("\nDebug: Checking index content...")
    print(f"Index has {len(index.docstore.docs)} documents")
    print(f"Index has {len(index.index_struct.nodes_dict)} nodes")

    # Save with explicit path
    save_path = "./saved_index"
    os.makedirs(save_path, exist_ok=True)
    index.storage_context.persist(persist_dir=save_path)

    # Verify saved content
    with open(os.path.join(save_path, "docstore.json"), "r") as f:
        print("\nDebug: Saved docstore content length:", len(f.read()))

except Exception as e:
    print(f"Error creating/saving index: {str(e)}")
    raise

end = time.time()
print(f"Total time taken: {end - start:.2f} seconds")