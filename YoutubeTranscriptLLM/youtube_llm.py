import os
import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv(find_dotenv())

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def create_faiss_database(video_url: str) -> FAISS:
    """Create a FAISS database from a YouTube video URL."""
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4) -> str:
    """Get a response from a query using the FAISS database and OpenAI model."""
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(temperature=0.4)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=textwrap.dedent("""
        You are a helpful assistant that can answer questions about YouTube videos
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        
        Format the text as required.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Pass input variables as a dictionary
    input_variables = {
        "question": query,
        "docs": docs_page_content
    }
    response_dict = chain.invoke(input=input_variables)

    # Access the 'text' key and replace newline characters
    response_text = response_dict.get('text', '')
    response = response_text.replace("\n", "")
    return response



def main():
    
    video_url = input("Enter Youtube Video link : ")
    db = create_faiss_database(video_url)

    query = input("Query? - ")
    response = get_response_from_query(db, query)
    # print(response)
    print(textwrap.fill(response, width=85))

if __name__ == "__main__":
    main()
