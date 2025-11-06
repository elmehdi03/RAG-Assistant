import streamlit as st
from rag_pipeline import ragpipeline


st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("🤖 RAG Assistant")

question = st.text_input("Posez votre question :")

if question:
    response = ragpipeline(question)
    st.subheader("Réponse :")
    st.write(response)
    st.balloons()