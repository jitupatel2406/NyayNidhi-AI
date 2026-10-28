from qdrant_client import QdrantClient

from app.config import QDRANT_PATH, COLLECTION_NAME
from app.rag.ingestion import extract_and_split
from app.rag.vector_store import create_vector_store
import streamlit as st
from app.rag.chain import get_rag_chain

st.set_page_config(
    page_title="NyayNidhi AI",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ NyayNidhi AI")
st.subheader("Your AI-powered BNS Legal Information Assistant")

with st.expander("📞 Emergency & Legal Aid Helplines"):
    st.markdown("**Emergency assistance — 112**")
    st.caption("For urgent police, fire, or medical emergencies.")
    
    st.markdown("**Free legal aid — NALSA: 15100**")
    st.caption("Contact the National Legal Services Authority for legal-aid assistance.")

    st.markdown("[Official emergency helpline information](https://112.gov.in/)")
    st.markdown("[Official NALSA legal-aid information](https://nalsa.gov.in/legal-aid/)")


st.warning(
    "This tool provides general legal information, not legal advice. "
    "For case-specific guidance, consult a qualified lawyer."
)
@st.cache_resource
def load_qa():
    client = QdrantClient(path=QDRANT_PATH)

    try:
        collection_exists = client.collection_exists(
            COLLECTION_NAME
        )
    finally:
        client.close()

    if not collection_exists:
        with st.spinner(
            "Preparing the legal database for the first time..."
        ):
            chunks = extract_and_split()
            create_vector_store(chunks)

    return get_rag_chain()

answer_question = load_qa()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View legal sources"):
                for source in message["sources"]:
                    st.markdown(f"**BNS PDF — Page {source['page']}**")
                    st.caption(source["content"])

question = st.chat_input("Ask a question about the Bharatiya Nyaya Sanhita...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the BNS and preparing an answer..."):
            try:
                result = answer_question(question)
                st.markdown(result["answer"])

                with st.expander("View legal sources"):
                    for source in result["sources"]:
                        st.markdown(f"**BNS PDF — Page {source['page']}**")
                        st.caption(source["content"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as error:
                st.error("Unable to generate an answer. Please try again.")
                st.caption(f"Technical details: {error}")
