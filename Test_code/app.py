import streamlit as st
from rag import Data, RAG


# Streamlit UI
st.image(
    "dowjones.jpg", width=600, caption="POWERING THE PROFESSIONAL WORLD"
)

query_text = st.text_area("Enter your query:")

if st.button("Submit Query"):
    if query_text:
        rag = RAG()
        with st.spinner("Processing your query..."):
            response_text = rag.query_rag(query_text)
        st.write(response_text)
    else:
        st.error("Please enter a query text.")
