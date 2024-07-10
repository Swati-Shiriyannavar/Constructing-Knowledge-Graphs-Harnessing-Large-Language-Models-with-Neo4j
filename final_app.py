import os
import streamlit as st
from py2neo import Graph, Node
from pdfminer.high_level import extract_text
from pyvis.network import Network
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph

# Set API key
os.environ["OPENAI_API_KEY"] = "OPENAI_KEY"

# Streamlit app
st.title("Knowledge Graph & Query System")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    text = extract_text("uploaded.pdf")

    # Write text to a file
    text_path = "extracted.txt"
    with open(text_path, 'w', errors='ignore') as text_file:
        text_file.write(text)

    # Load text from the file
    loader = TextLoader(text_path)
    documents = loader.load()

    # Define chunking strategy and split documents
    text_splitter = TokenTextSplitter(chunk_size=80, chunk_overlap=10)
    documents = text_splitter.split_documents(documents[:3])

    # Initialize LLM and transform documents to graph documents
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    llm_transformer = LLMGraphTransformer(llm=llm)
    langchain_graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Connect to Neo4j
    try:
        py2neo_graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "12345678"))
        st.success("Successfully connected to Neo4j!")
    except ValueError as e:
        st.error(f"Error connecting to Neo4j: {e}")

    # Create nodes in Neo4j
    for node in langchain_graph_documents[0].nodes:
        neo4j_node = Node("Concept", name=node.id, content='')
        py2neo_graph.create(neo4j_node)

    # Visualize the graph using PyVis
    net = Network(height="600px", width="100%", directed=True)
    for node in langchain_graph_documents[0].nodes:
        net.add_node(node.id, label=node.id, title=node.type)
    for relationship in langchain_graph_documents[0].relationships:
        net.add_edge(relationship.source.id, relationship.target.id, label=relationship.type, title=relationship.type)
    net.save_graph("mygraph.html")

    # Display the graph
    st.write("Generated Knowledge Graph")
    with open("mygraph.html", 'r', encoding='utf-8') as f:
        html_graph = f.read()
    st.components.v1.html(html_graph, height=600)

    # Process the main PDF file for similarity search
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Initialize OpenAI embeddings and connect to Neo4j
    embeddings = OpenAIEmbeddings()
    db = Neo4jVector.from_documents(docs, embeddings, url="bolt://localhost:7687", username="neo4j", password="12345678")

    # User query input
    st.write("Enter your query")
    user_query = st.text_input("Query")
    submit = st.button("Submit")

    if submit and user_query:
        docs_with_score = db.similarity_search_with_score(user_query, k=2)
    
        # Check if there are any results
        if docs_with_score:
            # Get the document with the highest score
            highest_score_doc, highest_score = max(docs_with_score, key=lambda x: x[1])
            # st.write("Query Result with Highest Score")
            # st.write(f"Score: {highest_score}")
            st.write(highest_score_doc.page_content)
        else:
            st.write("No results found")

