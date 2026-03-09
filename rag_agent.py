from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# ============================================================
# CONFIGURATION — Edit these values for your use case
# ============================================================
PDF_PATH = os.getenv("PDF_PATH", "your_document.pdf")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_collection")
DOCUMENT_DESCRIPTION = os.getenv("DOCUMENT_DESCRIPTION", "the uploaded PDF document")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 5))
# ============================================================


# LLM
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0  # minimize hallucination
)

# Embeddings (runs locally, no API key needed)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ── PDF Loading ──────────────────────────────────────────────
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

pdf_loader = PyPDFLoader(PDF_PATH)

try:
    pages = pdf_loader.load()
    print(f"✅ PDF loaded — {len(pages)} pages")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    raise

# ── Chunking ─────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
pages_split = text_splitter.split_documents(pages)
print(f"✅ Split into {len(pages_split)} chunks")

# ── ChromaDB Vector Store ────────────────────────────────────
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

try:
    # Load existing DB if available, otherwise create new
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("✅ Loaded existing ChromaDB vector store")
    else:
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        print("✅ Created new ChromaDB vector store")

except Exception as e:
    print(f"❌ Error setting up ChromaDB: {str(e)}")
    raise

# ── Retriever ────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVER_K}
)

# ── Tool ─────────────────────────────────────────────────────
@tool
def retriever_tool(query: str) -> str:
    """
    Searches and returns relevant information from the loaded PDF document.
    Use this tool to answer any questions about the document content.
    """
    docs = retriever.invoke(query)

    if not docs:
        return f"No relevant information found in {DOCUMENT_DESCRIPTION}."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)
tools_dict = {t.name: t for t in tools}

# ── Agent State ───────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ── System Prompt ─────────────────────────────────────────────
system_prompt = f"""
You are an intelligent AI assistant that answers questions based on {DOCUMENT_DESCRIPTION}.
Use the retriever tool to find relevant information. You can make multiple calls if needed.
Always cite the specific parts of the document you use in your answers.
If the answer is not found in the document, say so clearly — do not make up information.
"""

# ── Graph Nodes ───────────────────────────────────────────────
def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with current conversation state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm_with_tools.invoke(messages)
    return {'messages': [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"🔍 Calling tool: '{t['name']}' | Query: {t['args'].get('query', 'N/A')}")

        if t['name'] not in tools_dict:
            print(f"⚠️  Tool '{t['name']}' not found.")
            result = "Incorrect tool name. Please retry with an available tool."
        else:
            result = tools_dict[t['name']].invoke(t['args'])
            print(f"📄 Result length: {len(str(result))} chars")

        results.append(ToolMessage(
            tool_call_id=t['id'],
            name=t['name'],
            content=str(result)
        ))

    print("✅ Tool execution complete. Returning to LLM.")
    return {'messages': results}


def should_continue(state: AgentState):
    """Route to tool execution or end."""
    last = state['messages'][-1]
    return hasattr(last, 'tool_calls') and len(last.tool_calls) > 0

# ── Build Graph ───────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# ── Run Agent ─────────────────────────────────────────────────
def run_agent():
    print(f"\n=== RAG AGENT | Document: {DOCUMENT_DESCRIPTION} ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Your question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)
        print()


if __name__ == "__main__":
    run_agent()
