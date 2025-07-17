from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

## Load environment variables from the .env file
load_dotenv()

## Retrieve API keys from the environment variables
api_key= os.getenv("OPENAI_API_KEY")

## Retreive Chroma DB Direc Path
path= os.getenv("CHROMA_DB_DIREC_PATH")

## Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

## Custom prompt
template = """
You are a helpful assistant having a conversation with a human.

Given the following context from documents and conversation history, answer the human's question.

{context}

{chat_history}
Human: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template,
)

## Main answering function
def get_answer(question):
    
    # Load vector DB and embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma(
        persist_directory= path,
        embedding_function=embeddings
    )

    # Search documents
    docs = vectordb.similarity_search(question)

    # Set up LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=api_key)

    # Create a document QA chain 
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Run the chain
    result = qa_chain.invoke({
        "context": docs,
        "chat_history": memory.load_memory_variables({})["chat_history"],
        "question": question
    })

    # Update memory
    memory.save_context({"question": question}, {"output": result})

    print("ANSWER: ",result)
    return result

#get_answer('How to simplify the usage of the API?')
#get_answer('What does Climate imply?')
#get_answer('I want to know what the maxillary sinus is.')
get_answer('What does Poppler have?')
#get_answer('Which is a tool for curating and searching databases')
