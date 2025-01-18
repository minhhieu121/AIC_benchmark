from utils.retrieval_load_index import FusionRetriever, set_vector_weight, set_bm25_weight
from utils.evaluation import Evaluator
from utils.LLM_reranker import GeminiReranker
from tqdm import tqdm
from llama_index.core.schema import TextNode, Document
import os
# from flask import Flask, request, jsonify
import time
from llama_index.core import StorageContext, ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
import os
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from collections import defaultdict
import re
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd


load_dotenv()
MILVUS_URI = os.getenv('MILVUS_URI')
MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

model_name = 'gemini-1.5-flash-002'
description_dir = '/Users/albuscorleone/Documents/Schoolwork/Major/Paper AIC/AIC_benchmark/ta_prompt'

# Initialize Gemini LLM and Embedding models
llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=GOOGLE_API_KEY)

# Set the Gemini LLM as the default LLM in Settings
Settings.llm = llm
Settings.embed_model = embed_model

# Global variables to hold the loaded index and retrievers
loaded_index = None
query_engine = None
fusion_retriever = None

# Initialize Milvus vector store
vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
    collection_name=os.getenv('COLLECTION')
)

#Initialize LLMPredictor
# Create StorageContext with vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Create VectorStoreIndex
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

def load_index():
    global loaded_index, query_engine, fusion_retriever
    # print("Loading the saved index from Milvus...")
    # Load the document store from the saved directory
    print("Loading document store ...")
    docstore = SimpleDocumentStore.from_persist_dir("./saved_index")

    print("Loading vector store ...")
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    loaded_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    print(f"Number of documents in loaded index: {len(docstore.docs)}")

    # Create retrievers
    print("Creating retrievers for ownData_fusion...")
    vector_retriever = loaded_index.as_retriever(similarity_top_k=20)
    bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=20)

    # Create FusionRetriever and QueryEngine
    print("Creating FusionRetriever for ownData_fusion...")
    fusion_retriever = FusionRetriever([vector_retriever, bm25_retriever], similarity_top_k=20)
    # query_engine = RetrieverQueryEngine(retriever=fusion_retriever)

def perform_query(query):
    print(query)

    print(f"Received query: {query}")

    try:
        start_time = time.time()
        query_bundle = QueryBundle(query)
        final_results = fusion_retriever._retrieve(query_bundle)
        fused_results = ""

        top_k_vid = []

        for final_result in final_results:
            fused_results += f"Node Source: {final_result.node.metadata.get('source')}, Fused Score: {final_result.score}\n"
            top_k_vid.append(final_result.node.metadata.get('source'))
        end_time = time.time()
        print(f"Time taken to execute query: {end_time - start_time:.2f} seconds")

        return top_k_vid
    except Exception as e:
        print("Query error: ", {e})

def get_queries():
    print("Start reading")
    df = pd.read_csv('/Users/albuscorleone/Documents/Schoolwork/Major/Paper AIC/AIC_benchmark/MSRVTT_JSFUSION_test.csv', sep=';')
    print("End reading")

    
    df = df.sort_values(by='video_id')
    sentences = df['sentence'].tolist()
    video_ids = df['video_id'].tolist()

    print("DataFrame Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)

    return sentences, video_ids

if __name__ == '__main__':
    load_index()
    query_set, label_set = get_queries()
    for i, query in enumerate(query_set[:10]):
        print(f"{i}: {query}")
        print(label_set[i])

    num_samples = 199

    result = []

    for query in tqdm(query_set[0:num_samples]):
        top_k_vid = perform_query(query)

        top_k_description = {}
        for vid in top_k_vid:
            file_path = os.path.join(description_dir, vid +'.txt')
            with open(file_path, 'r') as f:
                vid_description = f.read()
                top_k_description.update({vid: vid_description})

        reranker = GeminiReranker(
            query=query, 
            top_k_vid=top_k_description, 
            model_name=model_name, 
            api_key=GOOGLE_API_KEY
        )

        top_k_vid = reranker.rerank()
        result.append(top_k_vid)
        
    metrics = Evaluator(label=label_set[0:num_samples], result=result)
    recall1, recall5, recall10 = metrics.perform_evaluation()
    print("Recall 1: ", recall1)
    print("Recall 5: ", recall5)
    print("Recall 10: ", recall10)

