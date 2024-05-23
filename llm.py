import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import CONSTANTS

llm = ChatOpenAI(
    openai_api_key = st.secrets[CONSTANTS.OPENAI_API_KEY]
    , model = st.secrets[CONSTANTS.OPENAI_MODEL]
)

embeddings = OpenAIEmbeddings (
    openai_api_key = st.secrets[CONSTANTS.OPENAI_API_KEY]
)