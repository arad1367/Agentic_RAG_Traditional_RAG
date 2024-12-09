from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType
from phi.playground import Playground, serve_playground_app

# 1. Traditional RAG
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://raw.githubusercontent.com/arad1367/UniLi_sources/main/WS_PejmanEbrahimi.pdf"],
    vector_db=PgVector(
        table_name="Place Branding in Tourist",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
knowledge_base.load(upsert=True)
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=False,
    markdown=True,
)
agent.print_response(
    "Hi, i want to make a professional abstract from a paper. Can you recommend best part of paper. "
    "I'd like to start with a introduction, then im thinking a methodology "
    "and finish with a IPMA matrix result",
    stream=True
)

# 2. Agentic RAG
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response(
    "Hi, i want to make a professional abstract from a paper. Can you recommend best part of paper. "
    "I'd like to start with a introduction, then im thinking a methodology "
    "and finish with a IPMA matrix result",
    stream=True
)

# 3. UI
rag_agent = Agent(
    name="RAG Agent",
    agent_id="rag-agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    storage=PgAgentStorage(table_name="rag_agent_sessions", db_url=db_url),
    instructions=[
        "Always search your knowledge base first and use it if available.",
        "Share the page number or source URL of the information you used in your response.",
        "If method benefits are mentioned, include them in the response.",
        "Important: Use tables where possible.",
    ],
    markdown=True,
)
app = Playground(agents=[rag_agent]).get_app()
if __name__ == "__main__":
    knowledge_base.load(upsert=True)
    serve_playground_app(app)