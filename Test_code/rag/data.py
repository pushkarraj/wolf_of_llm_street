from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import os, shutil

load_dotenv()


class Data:
    def __init__(self, data_path="./data", chroma_path="./chroma"):
        self.data_path = data_path
        self.chroma_path = chroma_path

    def _load_documents(self):
        document_loader = PyPDFDirectoryLoader(self.data_path)
        return document_loader.load()

    def _split_text(self, documents: list[Document]):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        return chunks

    def _save_to_chroma(self, chunks: list[Document]):

        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self.chroma_path
        )
        print(f"Saved {len(chunks)} chunks to {self.chroma_path}.")

    def _generate_data_store(self):

        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._save_to_chroma(chunks)


if __name__ == "__main__":
    data = Data()
    data._generate_data_store()
