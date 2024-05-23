import streamlit as st
from langchain_community.graphs import Neo4jGraph
import CONSTANTS

graph = Neo4jGraph(
    url = st.secrets[CONSTANTS.NEO4J_URI]
    , username = st.secrets[CONSTANTS.NEO4J_USERNAME]
    , password = st.secrets[CONSTANTS.NEO4J_PASSWORD]
)


