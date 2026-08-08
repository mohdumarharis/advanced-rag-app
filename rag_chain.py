"""RAG chain: prompt formatting + LLM invocation."""
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from llm_client import get_llm

_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the provided context below.

Instructions:
- If the answer is explicitly stated, provide it directly and concisely.
- If the context contains the answer spread across sentences, combine it into a clear answer.
- For acronyms like RAM-UR, RAM-UR-IRA, IRA, etc., look for the full phrase in parentheses or in the sentence immediately before/after the acronym.
- If the context contains absolutely no relevant information, say "I don't know based on these documents."
- If the context contains absolutely no relevant information, say 'I don't know based on these documents.' If the context clearly shows something does not exist (e.g., no code, no dataset), state that directly."
Context:
{context}

Question: {question}
Answer:"""

_prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)


def format_docs(docs: List) -> str:
    """Format retrieved documents into a single context string."""
    lines = []
    for d in docs:
        src = d.metadata.get("source", "?")
        slide = d.metadata.get("slide", "N/A")
        lines.append(f"[Source: {src} | Slide: {slide}]\n{d.page_content}")
    return "\n\n".join(lines)


def answer_from_docs(question: str, docs: List) -> str:
    """Generate an answer from documents that have already been retrieved."""
    context = format_docs(docs)
    messages = _prompt.format_messages(context=context, question=question)
    return get_llm().invoke(messages).content


def invoke(question: str, retriever) -> Tuple[str, List]:
    """
    Run the full RAG pipeline for a single question.
    Returns (answer_text, source_documents).
    """
    docs = retriever.retrieve(question)
    return answer_from_docs(question, docs), docs
