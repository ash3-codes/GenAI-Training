# rag/orchestrator.py

from llm.query_intelligence import QueryIntelligence
from llm.followup_rewriter import FollowUpRewriter
from retriever.vector_retriever import VectorRetriever
from retriever.gpt_reranker import AzureGPTReranker
from retriever.context_builder import ContextBuilder
from llm.answer_engine import AnswerEngine
from memory.conversation_memory import ConversationMemory


class HRPolicyOrchestrator:

    def __init__(self):

        # Intelligence Layer
        self.query_intelligence = QueryIntelligence()
        self.followup_rewriter = FollowUpRewriter()

        # Retrieval Layer
        self.retriever = VectorRetriever()
        self.reranker = AzureGPTReranker()
        self.context_builder = ContextBuilder()

        # Answer Engine
        self.answer_engine = AnswerEngine()

        # Memory
        self.memory = ConversationMemory()

        # Debug State
        self.raw_query = None
        self.normalized_query = None
        self.intent = None
        self.retrieved_docs = None
        self.reranked_docs = None
        self.context = None
        self.final_answer = None

    # ---------------------------------------------------------
    # Detect Short Follow-up Queries
    # ---------------------------------------------------------
    def is_followup(self, query: str) -> bool:
        return len(query.split()) <= 5

    # ---------------------------------------------------------
    # Small Talk Handler
    # ---------------------------------------------------------
    def handle_small_talk(self):

        if self.intent == "greeting":
            answer = "Hello, I am the HR Policy Bot. How can I assist you today?"

        elif self.intent == "identity":
            answer = "I am the HR Policy Bot designed to help you with company policy-related queries."

        else:
            answer = "I can assist with HR policy-related questions. Please ask about company policies."

        # Save memory
        self.memory.add_user_message(self.raw_query)
        self.memory.add_assistant_message(answer)

        return {
            "answer": answer,
            "sources": []
        }

    # ---------------------------------------------------------
    # Main Ask Method
    # ---------------------------------------------------------
    def ask(self, query: str):

        # Store Raw Query
        self.raw_query = query

        # -------------------------------
        # Query Intelligence
        # -------------------------------
        processed = self.query_intelligence.process(query)

        self.normalized_query = processed["normalized_query"]
        self.intent = processed["intent"]

        # -------------------------------
        # Intent Routing
        # -------------------------------
        if self.intent != "policy_lookup":
            return self.handle_small_talk()

        # -------------------------------
        # Follow-up Rewriting (Conditional)
        # -------------------------------
        if self.is_followup(self.normalized_query) and self.memory.get_recent_history():

            memory_context = self.memory.get_formatted_history()

            self.normalized_query = self.followup_rewriter.rewrite(
                self.normalized_query,
                memory_context
            )

        # -------------------------------
        # Retrieval (Wider)
        # -------------------------------
        self.retrieved_docs = self.retriever.retrieve(
            query=self.normalized_query,
            top_k=25
        )

        # -------------------------------
        # GPT Reranking
        # -------------------------------
        self.reranked_docs = self.reranker.rerank(
            query=self.normalized_query,
            documents=self.retrieved_docs,
            top_k=5
        )

        # -------------------------------
        # Weak Retrieval Detection
        # -------------------------------
        if not self.reranked_docs or len(self.reranked_docs) < 2:

            fallback_answer = (
                "I could not find sufficient information in the policy documents "
                "to answer your question accurately. Please rephrase or provide more details."
            )

            self.memory.add_user_message(self.raw_query)
            self.memory.add_assistant_message(fallback_answer)

            return {
                "answer": fallback_answer,
                "sources": []
            }

        # -------------------------------
        # Context Building
        # -------------------------------
        self.context = self.context_builder.build(self.reranked_docs)

        # -------------------------------
        # Answer Generation
        # -------------------------------
        response = self.answer_engine.generate(
            query=self.normalized_query,
            context=self.context
        )

        self.final_answer = response["answer"]

        # -------------------------------
        # Store Memory
        # -------------------------------
        self.memory.add_user_message(self.raw_query)
        self.memory.add_assistant_message(self.final_answer)

        return response
    

    # ---------------------------------------------------------
    # Debug State
    # ---------------------------------------------------------
    def debug_state(self):

        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "retrieved_docs_count": len(self.retrieved_docs) if self.retrieved_docs else 0,
            "reranked_docs_count": len(self.reranked_docs) if self.reranked_docs else 0,
        }