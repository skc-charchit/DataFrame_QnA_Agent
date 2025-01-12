import os
from langchain_community.llms import Ollama 
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the language model
llm = Ollama(
    model="llama3.2:3b",
    temperature=0.3,
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def get_answer(file_name, query):
    file_path = os.path.join(working_dir, file_name)
    
    try:
        # Load the documents
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        
        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(separator="\n", 
                                              chunk_size=1000, 
                                              chunk_overlap=200)
        
        text_chunks = text_splitter.split_documents(documents)
        
        # Embed the text chunks
        knowledge_base = FAISS.from_documents(text_chunks, embeddings)
        
        # Create a retriever from the knowledge base
        retriever = knowledge_base.as_retriever()
        
        # Create a RetrievalQA instance with combine_documents_chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # or "map_reduce", depending on your needs
            retriever=retriever,
            return_source_documents=True  # Optional parameter
        )
        
        # Invoke the chain with the query
        response = qa_chain({"query": query})
        
        # Print the entire response for debugging purposes
        print("Response:", response)  # Inspect the response structure
        
        # Attempt to return the answer, checking for different possible keys
        answer = response.get("answer") or response.get("result") or "No answer found."
    
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        answer = "An error occurred while processing the document."
    
    return answer
