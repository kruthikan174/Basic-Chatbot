import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model and master commands from Excel
model = SentenceTransformer('all-MiniLM-L6-v2')
commands_df = pd.read_excel("D:\\AI chatbot\\Streamlit_chatbot\\commands.xlsx")  # Load master command file
command_list = commands_df['command'].tolist()


# Helper function for conversational output
def conversational_response(command, column_name, result):
    return f"The {command} of column '{column_name}' is {result:.2f}."


# Natural error handling function
def handle_error(error_message):
    return f"Oops! I couldn't process your request. {error_message}"


# Function to dynamically detect command from user query
def process_query(user_input, model, commands_df):
    query_vector = model.encode([user_input])
    command_vectors = model.encode(command_list)  # Encode the commands from Excel
    similarities = cosine_similarity(query_vector, command_vectors)
    most_similar_index = np.argmax(similarities)

    # Access command and formula from Excel
    matched_command = commands_df.iloc[most_similar_index]['command']
    formula = commands_df.iloc[most_similar_index]['Formula']

    return matched_command, formula


# Streamlit App
st.title("Mathematical Chatbot 🧮")
st.markdown("<h3 style='text-align: center; color: white;'>Perform mathematical operations on your dataset</h3>",
            unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Dataset (.txt)", type="txt")

if uploaded_file:
    data = pd.read_csv(uploaded_file, delimiter="\t")  # Assuming tab-separated values
    st.write("Uploaded Dataset:")
    st.dataframe(data.head())  # Display the first few rows

    # Initialize chat history in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # User input and NLP model interaction
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about your dataset (e.g., 'What is the mean of column X?')", key="input")
        submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            try:
                # Process query using NLP model
                command, formula = process_query(user_input, model, commands_df)
                column_name = user_input.split()[-1]  # Assume column name is the last word

                if column_name not in data.columns:
                    raise ValueError(f"Column '{column_name}' is not in the dataset.")

                # Dynamically apply formula from Excel
                result = eval(f"data[column_name].{formula}()")  # Execute the formula dynamically

                # Conversational output
                response = conversational_response(command, column_name, result)

                # Store the chat history
                st.session_state['history'].append((user_input, response))

            except Exception as e:
                # Handle errors gracefully
                error_response = handle_error(str(e))
                st.session_state['history'].append((user_input, error_response))

    # Display chat history
    for user_input, response in st.session_state['history']:
        st.write(f"You: {user_input}")
        st.write(f"Bot: {response}")
