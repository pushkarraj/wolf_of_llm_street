from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class RAG:

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    def __init__(self, chroma_path="./chroma"):
        self.chroma_path = chroma_path
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_function,
        )

    def query_rag(self, query_text):

        results = self.db.similarity_search_with_relevance_scores(query_text, k=3)

        if len(results) == 0 or results[0][1] < 0.7:
            return "Unable to find matching results."

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI(model="gpt-4o", max_tokens=4078, temperature=0, top_p=0.0009)

        response_text = model.predict(prompt)

        # sources = [doc.metadata.get("source", None) for doc, _score in results]

        # formatted_response = f"Response: {response_text}\nSources: {sources}"

        return response_text
