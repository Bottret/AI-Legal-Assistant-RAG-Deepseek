from langchain_ollama import ChatOllama

import os
from vector_database import retrieve_docs as retrieve_filtered_docs
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# Step1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434"
)


# Step2: Retrieve Docs
def retrieve_docs(query, file_name):
    return retrieve_filtered_docs(query, file_name)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step3: Answer Question with Follow-Up Support
custom_prompt_template = """
Use the pieces of information provided in the context and previous conversation history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Previous Conversation:
{history}

Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query, history=""):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context, "history": history})
    return response

# Step4: Summarization Function
def summarize_document(documents):
    context = get_context(documents)
    summary_prompt = """
    Summarize the given legal document concisely while preserving key details.
    Provide a structured summary that highlights the most important points.
    
    Document:
    {context}
    
    Summary:
    """
    prompt = ChatPromptTemplate.from_template(summary_prompt)
    chain = prompt | llm_model
    return chain.invoke({"context": context})

# ####

def summarize_full_document(pdf_path):
    """
    Summarize the entire PDF (not just retrieved chunks) using map-reduce summarization.
    """

    # 1) Load full PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)

    # 3) MAP: summarize each chunk
    chunk_summaries = []

    map_prompt = ChatPromptTemplate.from_template("""
You are a legal assistant.
Summarize the following part of a legal document clearly and concisely.
Keep key numbers, dates, fees, penalties, and responsibilities.

Text:
{chunk}

Chunk Summary:
""")

    for chunk in chunks:
        chain = map_prompt | llm_model
        s = chain.invoke({"chunk": chunk.page_content})
        chunk_summaries.append(s.content)   # IMPORTANT

    # 4) REDUCE: combine chunk summaries
    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_template("""
You are a legal assistant.
Combine the following chunk summaries into one final structured summary.

Output format:
- Document Type
- Parties (if present)
- Term / Duration
- Payment / Rent
- Security Deposit
- Utilities
- Rules / Restrictions
- Termination / Notice
- Penalties / Fees
- Other Important Clauses

Chunk summaries:
{summaries}

Final Summary:
""")

    chain = reduce_prompt | llm_model
    final = chain.invoke({"summaries": combined})
    return final

# Step5: Generate Downloadable Report using ReportLab
def generate_report(user_queries, ai_responses):
    pdf_path = "AI_Lawyer_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Below is a record of your conversation with AI Lawyer.")

    y = 700
    max_width = 450
    line_height = 15

    for question, answer in zip(user_queries, ai_responses):
        # Question text (always string)
        q_text = str(question)
        q_lines = simpleSplit(f"Q: {q_text}", "Helvetica-Bold", 12, max_width)

        # Answer may be AIMessage OR string
        a_text = answer.content if hasattr(answer, "content") else str(answer)
        a_lines = simpleSplit(f"A: {a_text}", "Helvetica", 12, max_width)

        # Print Question
        c.setFont("Helvetica-Bold", 12)
        for line in q_lines:
            c.drawString(100, y, line)
            y -= line_height

        # Print Answer
        c.setFont("Helvetica", 12)
        for line in a_lines:
            c.drawString(100, y, line)
            y -= line_height

        y -= 20  # Extra space between Q&A

        # Prevent overflow
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    return pdf_path