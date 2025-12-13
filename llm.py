# llm.py

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence

GROQ_API_KEY = "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"

answer_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=180,
)

translation_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=180,
)

answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an expert assistant specialized in extracting answers from insurance policy documents.

    STRICT RULES:
    - Answer ONLY in **English**
    - Use only the information from CONTEXT
    - Write 30–40 words
    - Use formal policy language ("shall indemnify", "subject to", etc.)
    - If the context does not contain the answer, respond exactly:
      No relevant policy information found.

    CONTEXT:
    {context}

    QUESTION (respond in English only):
    {question}
    """
)

answer_chain = answer_prompt | answer_llm

def get_llm_answer(question: str, context: str) -> str:
    response = answer_chain.invoke({"context": context, "question": question})
    return response.content.strip()

lang_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Detect the language of this text. Answer ONLY one word:
    English, Hindi, or Marathi.

    TEXT:
    {text}
    """
)

lang_chain = lang_prompt | translation_llm


def detect_language(text: str) -> str:
    response = lang_chain.invoke({"text": text})
    lang = response.content.strip().lower()

    if "hindi" in lang:
        return "Hindi"
    if "marathi" in lang:
        return "Marathi"
    return "English"

to_eng_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Translate the following text into **English only**.
    Keep the meaning unchanged.
    TEXT:
    {text}
    """
)

to_eng_chain = to_eng_prompt | translation_llm


def translate_to_english(text: str) -> str:
    response = to_eng_chain.invoke({"text": text})
    return response.content.strip()

to_user_prompt = PromptTemplate(
    input_variables=["text", "lang"],
    template="""
    Translate the following English text into {lang}.
    Provide **only** the translated output:
    {text}
    """
)

to_user_chain = to_user_prompt | translation_llm

def translate_answer(text: str, target_language: str) -> str:
    english_clean = translate_to_english(text)
    if target_language == "English":
        return english_clean
    
    translated = to_user_chain.invoke({
        "text": english_clean,
        "lang": target_language
    }).content.strip()

    return f"""
**{target_language} Translation:**  
{translated}

**English Version:**  
{english_clean}
""".strip()
