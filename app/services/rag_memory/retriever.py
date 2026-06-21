import os
from typing import List, Dict, Any
# Use HuggingFace for robust, free local embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

class PostEventMemory:
    def __init__(self):
        # Defer initialization so FastAPI can load environment variables first
        self.embeddings = None
        self.vector_store = None
        self.index_name = "astram-memory"

    def _initialize_db(self):
        """
        Lazy initialization. Connects to Pinecone using a 768-dimension local embedding model.
        """
        if self.vector_store is None:
            # FIX: Bypassing Google's buggy embedding endpoint.
            # 'all-mpnet-base-v2' is a rock-solid local model that outputs exactly 768 dimensions
            # meaning it perfectly matches your existing Pinecone index!
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )

    def learn_from_feedback(self, event_id: str, event_cause: str, location: str, 
                            ai_plan: str, officer_feedback: str, rating: int) -> dict:
        """
        The Feedback Gatekeeper: Only stores operations rated 4 or 5 out of 5.
        """
        if rating < 4:
            print(f"Event {event_id} rated {rating}/5. Dropping from RAG memory to prevent bad learning.")
            return {"status": "ignored", "reason": "Rating below threshold"}

        print(f"Event {event_id} rated {rating}/5. Embedding to Cloud Pinecone...")

        # Initialize connection safely right before we need it
        self._initialize_db()

        # Create the searchable text
        searchable_content = f"Event Cause: {event_cause}. Location: {location}. Context: {officer_feedback}"

        # Store the tactical details in the metadata
        metadata = {
            "event_id": event_id,
            "cause": event_cause,
            "location": location,
            "rating": rating,
            "tactical_plan": ai_plan,
            "human_feedback": officer_feedback 
        }

        # Embed and upload to the cloud
        doc = Document(page_content=searchable_content, metadata=metadata)
        self.vector_store.add_documents([doc])
        
        return {"status": "success", "message": "Successfully embedded to Cloud Memory Bank."}

    def retrieve_similar_past_events(self, current_cause: str, current_location: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Fetches the top K most similar PAST successful operations from the cloud.
        Gracefully handles network drops or empty cloud indexes.
        """
        self._initialize_db()
        
        query_text = f"Event Cause: {current_cause}. Location: {current_location}."
        extracted_history = []
        
        try:
            # Perform similarity search in Pinecone
            
            results = self.vector_store.similarity_search(query_text, k=k)
            
            for doc in results:
                extracted_history.append({
                    "past_location": doc.metadata.get("location"),
                    "past_cause": doc.metadata.get("cause"),
                    "rating": doc.metadata.get("rating"),
                    "officer_lessons_learned": doc.metadata.get("human_feedback"),
                    "successful_plan_used": doc.metadata.get("tactical_plan")
                })
        except Exception as e:
            # Protects the API if Pinecone is down, keys are wrong, or index is missing
            print(f"⚠️ Pinecone Vector search bypassed (Network error or uninitialized index): {e}")
            return []
            
        return extracted_history

# Singleton instance
memory_bank = PostEventMemory()

# # backend/app/services/rag_memory/retriever.py

# import os
# from typing import List, Dict, Any
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.documents import Document

# class PostEventMemory:
#     def __init__(self):
#         # Defer initialization so FastAPI can load environment variables first
#         self.embeddings = None
#         self.vector_store = None
#         self.index_name = "astram-memory"

#     def _initialize_db(self):
#         """
#         Lazy initialization. Ensures OPENAI_API_KEY and PINECONE_API_KEY 
#         are fully loaded in the environment before connecting to the cloud.
#         """
#         if self.vector_store is None:
#             self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#             self.vector_store = PineconeVectorStore(
#                 index_name=self.index_name,
#                 embedding=self.embeddings
#             )

#     def learn_from_feedback(self, event_id: str, event_cause: str, location: str, 
#                             ai_plan: str, officer_feedback: str, rating: int) -> dict:
#         """
#         The Feedback Gatekeeper: Only stores operations rated 4 or 5 out of 5.
#         """
#         if rating < 4:
#             print(f"Event {event_id} rated {rating}/5. Dropping from RAG memory to prevent bad learning.")
#             return {"status": "ignored", "reason": "Rating below threshold"}

#         print(f"Event {event_id} rated {rating}/5. Embedding to Cloud Pinecone...")

#         # Initialize connection safely right before we need it
#         self._initialize_db()

#         # Create the searchable text
#         searchable_content = f"Event Cause: {event_cause}. Location: {location}. Context: {officer_feedback}"

#         # Store the tactical details in the metadata
#         metadata = {
#             "event_id": event_id,
#             "cause": event_cause,
#             "location": location,
#             "rating": rating,
#             "tactical_plan": ai_plan,
#             "human_feedback": officer_feedback 
#         }

#         # Embed and upload to the cloud
#         doc = Document(page_content=searchable_content, metadata=metadata)
#         self.vector_store.add_documents([doc])
        
#         return {"status": "success", "message": "Successfully embedded to Cloud Memory Bank."}

#     def retrieve_similar_past_events(self, current_cause: str, current_location: str, k: int = 2) -> List[Dict[str, Any]]:
#         """
#         Fetches the top K most similar PAST successful operations from the cloud.
#         Gracefully handles network drops or empty cloud indexes.
#         """
#         self._initialize_db()
        
#         query_text = f"Event Cause: {current_cause}. Location: {current_location}."
#         extracted_history = []
        
#         try:
#             # Perform similarity search in Pinecone
#             results = self.vector_store.similarity_search(query_text, k=k)
            
#             for doc in results:
#                 extracted_history.append({
#                     "past_location": doc.metadata.get("location"),
#                     "past_cause": doc.metadata.get("cause"),
#                     "rating": doc.metadata.get("rating"),
#                     "officer_lessons_learned": doc.metadata.get("human_feedback"),
#                     "successful_plan_used": doc.metadata.get("tactical_plan")
#                 })
#         except Exception as e:
#             # Protects the API if Pinecone is down, keys are wrong, or index is missing
#             print(f"⚠️ Pinecone Vector search bypassed (Network error or uninitialized index): {e}")
#             return []
            
#         return extracted_history

# # Singleton instance
# memory_bank = PostEventMemory()