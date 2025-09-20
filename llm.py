# llm.py

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

load_dotenv()

# Load API key
GROQ_API_KEY = "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"

# Initialize Groq LLaMA 3 model (via LangChain's OpenAI-compatible wrapper)
llm = ChatGroq(
    model_name="gemma2-9b-it",  # or "llama3-8b-8192" for lighter version
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=50
)

# Insurance-specific prompt template
prompt_template = PromptTemplate(
    template= """
    You are an expert AI assistant trained to extract and summarize detailed information from insurance policy documents.
    Your goal is to answer the user's QUESTION using only the provided CONTEXT. Follow these rules:
    Guidelines:
    Base your answer strictly on the CONTEXT.
    Aim for 30-40 words per answer
    Use formal policy language where applicable (e.g., “shall indemnify”, “subject to”, “provided that”).
    Each answer should be a complete, self-contained clause: detailed enough to capture eligibility, limits, waiting periods, conditions, and exceptions — but without becoming overly verbose or repetitive.
    If multiple distinct points are found, return them as separate items in the list.
    If the CONTEXT does not provide relevant information, return:
    Don't add any additional comments of your own stick to the answer itself.
    —
    📄 CONTEXT:
    {context}
    ❓ QUESTION:
    {question}
    —
    """
    )

# Build LangChain-compatible chain
def get_llm_answer(question: str, context: str) -> str:
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(context=context, question=question)