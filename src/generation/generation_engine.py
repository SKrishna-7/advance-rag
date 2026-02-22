from typing import List, Dict, Literal, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from logger.logger_config import get_logger

from langchain_groq import ChatGroq

from src.utils.config import ModelConfig

import dspy
from src.signatures.config_signatures import IntentClassifier
from src.signatures.config_signatures import CasualChat
from src.signatures.config_signatures import QueryRewriter
from src.signatures.config_signatures import DocumentGrader
from src.signatures.config_signatures import RAGAnswer


import os
logger = get_logger("Generator_LOG")



# --- ENGINE CLASS ---
base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class GenerationEngine:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initializes the Generator with a local Ollama model.
        """
        logger.info(f"Initializing Generator with model: {model_name}")
        self.config = ModelConfig()
        # model_name = model_name or self.config.MODEL_NAME
        # Main LLM for text generation
        
        print("Model Config : ",self.config.MODEL_NAME)

        if not model_name:
            if self.config.LLM_PROVIDER == "ollama":
                model_name = self.config.OLLAMA_MODEL_NAME
            else:
                model_name = self.config.MODEL_NAME
                
        if self.config.LLM_PROVIDER == "ollama":
            target_model = model_name or self.config.OLLAMA_MODEL_NAME
            # Use unified dspy.LM with the "ollama/" prefix
            self.lm = dspy.LM(
                f"ollama_chat/{target_model}",
                api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=self.config.TEMP,
                max_tokens=2000,
                cache=False
            )
        else:
            target_model = model_name or self.config.MODEL_NAME
            # Use unified dspy.LM with the "groq/" prefix
            self.lm = dspy.LM(
                f"groq/{target_model}",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=self.config.TEMP,
                cache=False
            )

        dspy.settings.configure(lm = self.lm)
        print("Dspy Initialized ...")

       

    # initilizing DSpy Signatures
        self.router_module = dspy.Predict(IntentClassifier)
        self.chat_modules = dspy.Predict(CasualChat)
        self.grader_module = dspy.Predict(DocumentGrader)
        self.rewrite_module = dspy.Predict(QueryRewriter)

        self.rag_module = dspy.ChainOfThought(RAGAnswer)

        
    # --- PUBLIC METHODS ---

    def check_intent(self, query: str) -> str:
        """Determines if query is CHAT or SEARCH."""
        try:
            result = self.router_module(question = query)
            intent = result.intent.strip().upper()

            if "SEARCH" in intent: return "SEARCH"
            return "CHAT"
        except Exception as e:
            logger.warning(f"Router failed: {e}")
            return "SEARCH" # Default safe option

    def chat_casually(self, query: str) -> str:
        """Handles casual conversation."""
        return self.chat_modules(question=query).response

    def grade_document(self, question: str, document: str) -> str:
        """Determines if a document is relevant (yes/no)."""
        try:
            result = self.grader_module(question = question , document = document)
            score = result.binary_score.strip().lower()
            print("Score : ",score)
            if "yes" in score : return "yes"
            return "no"
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            return "yes" # Default to keeping it if grading fails

    def rewrite_query(self, question: str) -> str:
        """Rewrites the question for better retrieval."""
        result = self.rewrite_module(question = question)
        return result.improved_question

    def generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[str] = []) -> str:
        if not retrieved_docs:
            return "I could not find any relevant documents to answer your question."

        context_str = self._format_context(retrieved_docs)
        history_str = "\n".join(chat_history) if chat_history else "No previous history."
        
        try:
            result = self.rag_module(
                context=context_str,
                question=query,
                chat_history=history_str
            )
            return result.answer
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating the response."

    def stream_generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[str] = []):
        """Streams the RAG response token by token."""
        if not retrieved_docs:
            yield "I could not find any relevant documents to answer your question."
            return

        context_str = self._format_context(retrieved_docs)
        history_str = "\n".join(chat_history) if chat_history else "No previous history."
        
        try:
          
          full_answer = self.generate_answer(
              query,
              retrieved_docs,
              chat_history

          )

          print("\n Full anwer " , full_answer)

          words = full_answer.split(" ")
        
          for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "Error generating response."

    
    def native_stream_answer(self, query: str, retrieved_docs: List[Document], chat_history: List[str] = []):
        """Bulletproof Simulated Streaming for Interview Day."""
        if not retrieved_docs:
            yield "I could not find any relevant documents to answer your question."
            return

        context_str = self._format_context(retrieved_docs)
        history_str = "\n".join(chat_history) if chat_history else "No previous history."
        
        try:
            logger.info("Executing DSPy Generation...")
            
            # 1. Generate the full answer normally (Safe, blocking call)
            result = self.rag_module(context=context_str, question=query, chat_history=history_str)
            full_answer = result.answer
            
            # 2. Simulate the stream back to the UI
            words = full_answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                
        except Exception as e:
            logger.error(f"DSPy Execution Failed: {e}")
            yield f"I encountered an error while generating the answer: {str(e)}"


    def _format_context(self, docs: List[Document]) -> str:
        """Helper to format docs into a string with metadata."""
        formatted_text = ""
        for doc in docs:
            source = doc.metadata.get("filename", "Unknown File")
            page = doc.metadata.get("page", "?")
            
            formatted_text += f"--- Source: {source} (Page {page}) ---\n"
            formatted_text += f"{doc.page_content}\n\n"
            
        return formatted_text
    
        