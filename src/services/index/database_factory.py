from typing import Union

from config import GlobalConfig

from .faiss import FAISSDatabase
from .milvus import MilvusDatabase


class DatabaseFactory:
    """Factory class to create database instances based on configuration or parameters"""

    @staticmethod
    def create_database(
        database_type: str = None,
        collection_name: str = "default_collection",
        do_overwrite: bool = False,
        work_dir: str = None,
    ) -> Union[FAISSDatabase, MilvusDatabase]:
        """
        Create a database instance based on the specified type

        Args:
            database_type: "faiss" or "milvus". If None, uses config default
            collection_name: Name of the collection
            do_overwrite: Whether to overwrite existing collection
            work_dir: Working directory for FAISS (ignored for Milvus)

        Returns:
            Database instance
        """
        if database_type is None:
            database_type = GlobalConfig.get("webui", "database") or "faiss"

        database_type = database_type.lower()

        if database_type == "faiss":
            return FAISSDatabase(collection_name, do_overwrite, work_dir)
        elif database_type == "milvus":
            if do_overwrite:
                MilvusDatabase.start_server()
            return MilvusDatabase(collection_name, do_overwrite)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

    @staticmethod
    def get_supported_databases():
        """Get list of supported database types"""
        return ["faiss", "milvus"]
