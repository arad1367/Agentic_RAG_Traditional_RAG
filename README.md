# Agentic_RAG_Traditional_RAG

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Code Setup](#code-setup)
    - [Run Application](#run-application)
4. [Docker Management](#docker-management)
5. [Example Query](#example-query)
6. [License](#license)

A comparison implementation of Traditional RAG and Agentic RAG using phidata framework.

## Prerequisites

- Python 3.8+
- Docker
- OpenAI API Key

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install phidata openai sqlalchemy 'psycopg[binary]' pgvector 'fastapi[standard]' pypdf

# Set OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Start pgvector container
docker run -d \
    -e POSTGRES_DB=ai \
    -e POSTGRES_USER=ai \
    -e POSTGRES_PASSWORD=ai \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v pgvolume:/var/postgresql/data \
    -p 5532:5432 \
    --name pgvector \
    phidata/pgvector:16
```

## Usage

### Code Setup
Create `app.py`:

```python
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType
from phi.playground import Playground, serve_playground_app

# Database configuration
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFUrlKnowledgeBase(
    urls=["your_pdf_url_here"],
    vector_db=PgVector(
        table_name="knowledge_store",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
knowledge_base.load(upsert=True)

# Traditional RAG
traditional_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=False,
    markdown=True,
)

# Agentic RAG
agentic_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

# UI Setup
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
```

### Run Application
```bash
python app.py
```

## Docker Management

```bash
# Check running containers
docker ps

# Stop specific container
docker stop containerID

# Stop all containers
docker stop $(docker ps -q)

# Delete all containers
docker ps -a -q | xargs -r docker rm
```

## Example Query
```python
query = """
Hi, i want to make a professional abstract from a paper. 
Can you recommend best part of paper. I'd like to start with a introduction, 
then im thinking a methodology and finish with a IPMA matrix result
"""

# Traditional RAG
traditional_agent.print_response(query, stream=True)

# Agentic RAG
agentic_agent.print_response(query, stream=True)
```

## License
MIT

---
For more details or issues, please refer to the phidata documentation.