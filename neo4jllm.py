import os
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

# Connect to Neo4j
try:
    graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "12345678"))
    print("Successfully connected to Neo4j!")
except ValueError as e:
    print(f"Error connecting to Neo4j: {e}")

# File paths
pdf_file = "Financial_report_edited.pdf"
text_path = "extracted.txt"

# Extract text from PDF
text = extract_text(pdf_file)

# Write text to a file
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

# Print nodes and relationships for verification
# print(f"Nodes: {langchain_graph_documents[0].nodes}")
# print(f"Relationships: {langchain_graph_documents[0].relationships}")

# Create nodes in Neo4j
py2neo_graph = Graph(uri="bolt://localhost:7687", auth=("neo4j", "12345678"))
for node in langchain_graph_documents[0].nodes:
    neo4j_node = Node("Concept", name=node.id, content='')
    py2neo_graph.create(neo4j_node)

# Visualize the graph using PyVis
net = Network(height="600px", width="100%", directed=True)
for node in langchain_graph_documents[0].nodes:
    net.add_node(node.id, label=node.id, title=node.type)
for relationship in langchain_graph_documents[0].relationships:
    net.add_edge(relationship.source.id, relationship.target.id, label=relationship.type, title=relationship.type)
net.save_graph("mygraph1.html")

# Process the main PDF file
pdf_file = "Financial_report.pdf"
text = extract_text(pdf_file)
text_path = "extracted_text.txt"
with open(text_path, 'w', errors='ignore') as text_file:
    text_file.write(text)

# Load and split documents
loader = TextLoader(text_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings and connect to Neo4j
embeddings = OpenAIEmbeddings()
db = Neo4jVector.from_documents(docs, embeddings, url="bolt://localhost:7687", username="neo4j", password="12345678")

# Perform similarity search
query = "What are AI/ML applications designed to improve product quality?"
docs_with_score = db.similarity_search_with_score(query, k=2)
print(docs_with_score)
