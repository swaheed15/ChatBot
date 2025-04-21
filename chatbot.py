import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate


# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Add validation for missing API key
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your environment variables.")

from pydantic import ValidationError, validate_call


@validate_call
def foo(a: int):
    return a


try:
    foo()
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'missing_argument'
    # > 'missing_argument' is a subclass of 'validation_error'  
# Initialize models with API key
llama_model = ChatGroq(api_key=groq_api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
compound_beta_model = ChatGroq(api_key=groq_api_key, model="compound-beta-mini")
deepseek_model = ChatGroq(api_key=groq_api_key, model="deepseek")
mistral_saba_model = ChatGroq(api_key=groq_api_key, model="mistral-saba-24b")

# Streamlit App Configuration
st.set_page_config(
    page_title="Memory Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Memory Chatbot")
st.write("This is a simple chatbot that remembers the conversation history.")
st.write("It uses the Groq API to generate responses and stores the conversation in memory.")

# Sidebar for model selection, temperature, and max tokens
model_name = st.sidebar.selectbox("Select a model", [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "compound-beta-mini",
    "mistral-saba-24b",
])

st.sidebar.write("Select a model to use for the chatbot.")
st.sidebar.write("Adjust the temperature and max tokens to control the response generation.")
st.sidebar.write("Temperature controls the randomness of the response.")
st.sidebar.write("Max tokens limit the length of the response.")

# Update the load_model function to handle groq.NotFoundError specifically
def load_model(model_name):
    try:
        if model_name == "meta-llama/llama-4-scout-17b-16e-instruct":
            return llama_model
        elif model_name == "compound-beta-mini":
            return compound_beta_model
        elif model_name == "mistral-saba-24b":
            return mistral_saba_model
        else:
            raise ValueError(f"Model '{model_name}' not recognized. Please select a valid model.")
    except groq.NotFoundError as e:
        st.error(f"Model '{model_name}' not found or access denied: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {str(e)}")
        return None

# Update the selected_model assignment to handle None
selected_model = load_model(model_name)
if not selected_model:
    st.stop()  # Stop execution if the model fails to load

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max tokens", 1, 4096, 512)

# Memory and conversation setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Build conversation chain with the selected model
conv = ConversationChain(
    llm=selected_model,
    memory=st.session_state.memory,
    verbose=False,
)

# Handling User input and predicting response
user_input = st.text_input("You:")

if user_input:
    st.session_state.history.append(("user", user_input))

    # Generate AI response
    response = conv.predict(input=user_input)
    st.session_state.history.append(("assistant", response))

    # Display conversation
    for role, text in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
else:
    st.warning("Please enter a message before submitting.")

