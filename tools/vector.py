import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings

neo4jvector = Neo4jVector.from_existing_index(
    embeddings
    , url = st.secrets['NEO4J_URI']
    , username = st.secrets['NEO4J_USERNAME']
    , password = st.secrets['NEO4J_PASSWORD']
    , index_name = 'moviePlots'
    , node_label = 'Movie'
    , text_node_property = 'plot'
    , emdbedding_node_property = 'plotEmbedding'
    , retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]

# tag::qa[]
kg_qa = RetrievalQA.from_chain_type(
    llm,                  # <1>
    chain_type="stuff",   # <2>
    retriever=retriever,  # <3>
)
# end::qa[]

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = kg_qa({"question": prompt})

    return response['answer']
# end::generate-response[]



## The `kg_qa` can now be registered as a tool within the agent.

# tag::importtool[]
from langchain.tools import Tool
# end::importtool[]

# tag::importkgqa[]
from tools.vector import kg_qa
# end::importkgqa[]

# tag::tool[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
        ),
    Tool.from_function(
        name="Vector Search Index",  # <1>
        description="Provides information about movie plots using Vector Search", # <2>
        func = kg_qa, # <3>
        return_direct=True
    )
]
# end::tool[]
