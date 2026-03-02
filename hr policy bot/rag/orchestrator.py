# rag/orchestrator.py

from llm.query_intelligence import QueryIntelligence
from retriever.vector_retriever import VectorRetriever
from retriever.gpt_reranker import AzureGPTReranker
from retriever.context_builder import ContextBuilder
from llm.answer_engine import AnswerEngine


class HRPolicyOrchestrator:

    def __init__(self):

        # Intelligence Layer
        self.query_intelligence = QueryIntelligence()

        # Retrieval Layer
        self.retriever = VectorRetriever()
        self.reranker = AzureGPTReranker()
        self.context_builder = ContextBuilder()

        # Answer Engine
        self.answer_engine = AnswerEngine()

        # State
        self.raw_query = None
        self.normalized_query = None
        self.intent = None
        self.retrieved_docs = None
        self.reranked_docs = None
        self.context = None
        self.final_answer = None

    # ---------------------------------------------------------
    # Small Talk Handler
    # ---------------------------------------------------------
    def handle_small_talk(self):

        if self.intent == "greeting":
            return {
                "answer": "Hello, I am the HR Policy Bot. How can I assist you today?",
                "sources": []
            }

        if self.intent == "identity":
            return {
                "answer": "I am the HR Policy Bot designed to help you with company policy-related queries.",
                "sources": []
            }

        return {
            "answer": "I can assist with HR policy-related questions. Please ask about company policies.",
            "sources": []
        }

    # ---------------------------------------------------------
    # Main Ask Method
    # ---------------------------------------------------------
    def ask(self, query: str):

        # Store Raw Query
        self.raw_query = query

        # Query Intelligence Processing
        processed = self.query_intelligence.process(query)

        self.normalized_query = processed["normalized_query"]
        self.intent = processed["intent"]

        # Intent Routing
        if self.intent != "policy_lookup":
            return self.handle_small_talk()

        # Retrieval 
        self.retrieved_docs = self.retriever.retrieve(
            query=self.normalized_query,
            top_k=15
        )

        # Reranking
        self.reranked_docs = self.reranker.rerank(
            query=self.normalized_query,
            documents=self.retrieved_docs,
            top_k=5
        )

        # Context Building
        self.context = self.context_builder.build(self.reranked_docs)

        # Grounded Answer Generation
        response = self.answer_engine.generate(
            query=self.normalized_query,
            context=self.context
        )

        self.final_answer = response["answer"]

        return response

    # ---------------------------------------------------------
    # Optional Debug State
    # ---------------------------------------------------------
    def debug_state(self):

        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "retrieved_docs_count": len(self.retrieved_docs) if self.retrieved_docs else 0,
            "reranked_docs_count": len(self.reranked_docs) if self.reranked_docs else 0,
        }