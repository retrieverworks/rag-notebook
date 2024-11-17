import streamlit as st
import requests

# API endpoints
UPLOAD_ENDPOINT = "http://127.0.0.1:8000/upload-pdf/"
QUERY_ENDPOINT = "http://127.0.0.1:8000/query/"

# -----------------------------------------
# Streamlit UI
# -----------------------------------------

# Title and Description
st.title("PDF Query Interface")
st.markdown("""
This application allows you to:
1. Upload a PDF file to extract and ingest its content.
2. Query the ingested content using a language model (Ollama or ChatGPT).
""")

# File Upload Section
st.header("Step 1: Upload a PDF")
uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])

if uploaded_file is not None:
    # Display the filename
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Upload to the FastAPI server
    with st.spinner("Uploading and processing the PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files)

    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully!")
    else:
        st.error("Failed to upload and process the PDF.")
        st.error(response.text)

# Query Section
st.header("Step 2: Query the PDF Content")

query = st.text_input("Enter your query:")
model_type = st.selectbox(
    "Select the model type:",
    options=["ollama", "chatgpt"],
    help="Choose the model to use for answering your query."
)

if st.button("Submit Query"):
    if not query:
        st.error("Please enter a query before submitting.")
    else:
        # Query the FastAPI server
        with st.spinner("Processing your query..."):
            data = {"query": query, "model_type": model_type}
            response = requests.post(QUERY_ENDPOINT, data=data)

        if response.status_code == 200:
            result = response.json().get("response", "No response received.")
            st.success("Response from the model:")
            st.write(result)
        else:
            st.error("Failed to process your query.")
            st.error(response.text)
