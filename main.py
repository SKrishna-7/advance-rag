import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.graph.rag_graph import RAGGraph
from src.ingestion.data_ingestion import DataIngestionPipeline
from src.utils.config import DataIngestionConfig

from src.utils.config import ModelConfig
from src.evaluator.rag_evaluator import RAGEvaluator

from dotenv import load_dotenv

load_dotenv()
import os


def main():
    print("\n" + "="*60)
    print("🤖 STRICT RAG SYSTEM (Knowledge Base Only)")
    print("="*60)

    evaluator = RAGEvaluator(model_name=ModelConfig.MODEL_NAME)

    # 1. Ingestion (Idempotent - skips if already done)
    print("[1/2] Checking Document Store...")
    DataIngestionPipeline(DataIngestionConfig()).run(r"docs\raw")

    # 2. Initialize Graph
    print("[2/2] Compiling Graph...")
    rag_bot = RAGGraph()
    
    print("\n✅ SYSTEM READY! (Type 'exit' to quit)")

    # 3. Session Config (For Memory)
    # In a real app, this changes per user
    config = {"configurable": {"thread_id": "user_session_1"}}

    while True:
        try:
            user_query = input("\n🔍 Query: ").strip()
            if user_query.lower() in ["exit", "quit"]: break
            if not user_query: continue

            # --- THE MAGIC ---
            # No intent checking. Just straight execution.
            inputs = {"question": user_query}
            
            # The graph now handles the entire Retrieve -> Generate loop
            result = rag_bot.app.invoke(inputs, config=config)

            print("\n🤖 Answer:", result["answer"])
            
            # Optional: Show citations/docs if available
            if result.get("documents"):
                print(f"\n[Used {len(result['documents'])} source documents]")

            score = evaluator.evaluate(
        question=user_query,
        answer=result["answer"],
        documents=result["documents"]
)

            print(f"Faithfulness: {score['faithfulness']}")
            print(f"Relevance: {score['relevance']}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()