from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    llm_chat_model: str = "deepseek-ai/deepseek-v4-flash"
    llm_embedding_model: str = "nvidia/nv-embedqa-e5-v5"

    pg_host: str = "172.16.3.20"
    pg_port: int = 5432
    pg_database: str = "assistant"
    pg_user: str = "assistant"
    pg_password: str = "ecBihPAddt3B6reR"

    agent_short_term_size: int = 20
    agent_top_k_chunks: int = 5
    agent_top_k_memories: int = 3
    agent_similarity_threshold: float = 0.3
    agent_chunk_size: int = 500
    agent_chunk_overlap: int = 50

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_database} user={self.pg_user} "
            f"password={self.pg_password}"
        )


settings = Settings()
