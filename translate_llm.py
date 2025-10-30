import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


DATA_PATH = "courses_data/"
VECTOR_STORE_PATH = "vector_store"


def get_conversational_rag_chain(retriever):
    """
    Creates the main conversational RAG chain.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_system_prompt = (
        "You are a helpful student support assistant for a university. "
        "Use the following retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep your answers concise and helpful.\n\n"
        "You need to talk like an support team member talks on call regarding queries , take the retrieved content as context and talk as humanly as possible"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain

def main():
    """
    Main function to run the chat bot.
    """

    # Load the Vector database thsat we have made
    print("Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    
    conversational_rag_chain = get_conversational_rag_chain(retriever)

    print("\n----------------------------------------------------")
    print(" Student Support Bot is ready! Ask me anything about the courses.")
    print("Type 'exit' to end the chat.")
    print("----------------------------------------------------")

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print(" Goodbye!")
            break

        # Invoke the chain with the input and history
        response = conversational_rag_chain.invoke(
            {"input": user_input, "chat_history": chat_history}
        )
        
        # Print the answer
        print(f"Bot: {response['answer']}")
        
        # Update the chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))

if __name__ == "__main__":
    main()