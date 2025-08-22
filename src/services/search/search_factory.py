from typing import Union

from config import GlobalConfig

from .faiss_searcher import Searcher as FAISSSearcher
from .milvus_searcher import Searcher as MilvusSearcher


class SearchFactory:
    """Factory class to create search instances based on configuration"""

    @staticmethod
    def create_searcher(
        collection_name: str,
        database_type: str = None,
    ) -> Union[FAISSSearcher, MilvusSearcher]:
        """
        Create a searcher instance based on the specified database type

        Args:
            collection_name: Name of the collection to search
            database_type: "faiss" or "milvus". If None, uses config default

        Returns:
            Searcher instance
        """
        if database_type is None:
            database_type = GlobalConfig.get("webui", "database") or "faiss"

        database_type = database_type.lower()

        if database_type == "faiss":
            return FAISSSearcher(collection_name)
        elif database_type == "milvus":
            return MilvusSearcher(collection_name)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

    @staticmethod
    def get_supported_databases():
        """Get list of supported database types"""
        return ["faiss", "milvus"]
