# Every other file in this project imports from here.

# pydantic is used for reading environment variables and validating them.
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# resolve the project root directory
# this is used for loading the .env file and for other file paths in the project.
ROOT_DIR = Path(__file__).parent.parent.parent # config/settings.py -> config/ -> financial_assistant/

class Settings(BaseSettings):
    # OpenAI API key
    OPENAI_API_KEY: str = Field(..., description="Your OpenAI API key")

    # llm model to use
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model name")

    # Optional: point at a different OpenAI-COMPATIBLE API instead of
    # OpenAI itself — e.g. Groq's free tier (https://api.groq.com/openai/v1).
    # Leave unset (None) to use OpenAI directly, unchanged from before.
    # When set, OPENAI_API_KEY should hold that provider's key instead —
    # the field name stays OPENAI_API_KEY either way since ChatOpenAI's
    # api_key parameter is what actually gets sent as the bearer token,
    # regardless of which provider is on the other end of base_url.
    LLM_BASE_URL: str | None = Field(
        default=None,
        description="Custom base URL for an OpenAI-compatible API (e.g. Groq). None = use OpenAI directly.",
    )

    # MAX_TOKENS for the LLM response
    LLM_MAX_TOKENS: int = Field(default=1024, description="Max tokens in LLM response")

    # LLM temperature for response generation
    # 0.0 = deterministic (same input → same output).
    LLM_TEMPERATURE: float = Field(default=0.0, description="LLM sampling temperature")

    # embedding model to use for vector store
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )

    # chunking settings for documents added to vector store
    CHUNK_SIZE: int = Field(default=500, description="Characters per document chunk")
    CHUNK_OVERLAP: int = Field(default=100, description="Overlap between chunks")
 
    # Vector store settings - using ChromaDB for this project
    CHROMA_PATH: Path = Field(
        default=ROOT_DIR / "data" / "processed" / "chroma_db",
        description="Where ChromaDB persists its data"
    )
    # name of the collection in ChromaDB where document embeddings are stored
    CHROMA_COLLECTION_NAME: str = Field(
        default="financial_docs",
        description="ChromaDB collection name"
    )

    # retrieval settings for the RAG pipeline
    RETRIEVAL_TOP_K: int = Field(default=5, description="Number of chunks to retrieve")

    # If True, response_generator.generate() uses hybrid (dense + BM25)
    # retrieval instead of dense-only semantic search. Defaults to False
    # so existing behavior is unchanged until explicitly opted into via
    # .env — flip this once you've compared retrieve() vs retrieve_hybrid()
    # on your own documents and are happy with the results.
    USE_HYBRID_RETRIEVAL: bool = Field(
        default=False,
        description="Use hybrid (dense+BM25) retrieval instead of dense-only",
    )

    # Data paths 
    RAW_DATA_DIR: Path = Field(
        default=ROOT_DIR / "data" / "raw",
        description="Where raw input documents live"
    )
    PROCESSED_DATA_DIR: Path = Field(
        default=ROOT_DIR / "data" / "processed",
        description="Where cleaned/processed data is saved"
    )

    # logging settings 
    # options are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity level")

    class Config:
        # Tell pydantic-settings to look for a .env file at the project root.
        # Variables in .env override defaults set above.
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"

# singleton pattern for settings - ensures we only load and parse environment variables once, only one instance of Settings is created and shared across the app.
settings = Settings()