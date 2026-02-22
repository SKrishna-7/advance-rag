import dspy

class IntentClassifier(dspy.Signature):
    """You are an intent classifier. Determine if the user's query is 'CHAT' 
    (greetings, small talk, compliments) or 'SEARCH' (questions needing information, context, or facts).\n
            Return ONLY one word: 'CHAT' or 'SEARCH'."""
    
    question = dspy.InputField()
    intent = dspy.OutputField(des="Strictly 'CHAT' or 'SEARCH' ")


class CasualChat(dspy.Signature):
    """
    You are a helpful AI assistant. Respond politely to the user's greeting 
        or conversational query. Do not hallucinate technical information.
    """

    question = dspy.InputField()
    response = dspy.OutputField()


class DocumentGrader(dspy.Signature):
    """Assess the relevance of a retrieved document to a user question. 
    If the document contains keywords or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no'."""

    document = dspy.InputField(desc="Retrieved document context")
    question = dspy.InputField(desc="User question")
    binary_score = dspy.OutputField(desc="Strictly 'yes' or 'no'")



class QueryRewriter(dspy.Signature):
    """Convert an input question to a better version optimized for vectorstore retrieval. 
    Look at the input and reason about the underlying semantic intent."""

    question = dspy.InputField(desc="Initial Question")
    improved_question = dspy.OutputField(desc="Formulate an improved question")



class RAGAnswer(dspy.Signature):
    """You are a technical assistant. Answer the user's question using ONLY the provided context snippets. 
    If the answer is not in the context, say so. CITATION FORMAT: [Source: filename (Page X)]"""

    chat_history = dspy.InputField(desc="Previous conversation history")
    context = dspy.InputField(desc="Retrieved document snippets")
    question = dspy.InputField(desc="User question")
    answer = dspy.OutputField()
