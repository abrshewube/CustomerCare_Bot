import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['OPENAI_API_KEY'] =os.getenv('OPENAI_API_KEY')

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="Customer_Service_Assistants.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# customer_messsage = """
# Hello, 

# I need your help cancelling an order I made.

# Thank you.
# """

# results = retrieve_info(customer_messsage)
# print(results)

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, tone of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
# def generate_response(message):
#     best_practice = retrieve_info(message)
#     response = chain.run(message=message, best_practice=best_practice)
#     return response

# Initialize conversation history
conversation_history = []
# breakpoint()
def update_conversation_history(user_message, bot_response):
    # Limit the history to the last 5 messages (adjust as needed)
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": bot_response})
    if len(conversation_history) > 10:
        conversation_history.pop(0)
        conversation_history.pop(0)

def get_contextual_input(conversation_history, new_message):
    history_str = ""
    for entry in conversation_history:
        role = entry["role"]
        content = entry["content"]
        history_str += f"{role}: {content}\n"
    history_str += f"user: {new_message}\n"
    return history_str

def generate_response(message):
    contextual_input = get_contextual_input(conversation_history, message)
    best_practice = retrieve_info(contextual_input)
    response = chain.run(message=contextual_input, best_practice=best_practice)
    update_conversation_history(message, response)
    return response


# Define global conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

def main():
    st.set_page_config(page_title="Customer Response Generator", page_icon=":bird:")
    st.header("Customer Response Generator :bird:")

    message = st.text_area("Customer message")
    if message:
        with st.spinner("Generating best practice message..."):
            try:
                # Get context-aware response
                contextual_input = get_contextual_input(st.session_state["conversation_history"], message)
                result = generate_response(contextual_input)
                
                # Update conversation history in session state
                st.session_state["conversation_history"].append({"role": "user", "content": message})
                st.session_state["conversation_history"].append({"role": "assistant", "content": result})
                
                st.info(result)
            except Exception as e:
                st.error(f"Error generating best practice message: {e}")

if __name__ == '__main__':
    main()