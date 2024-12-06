#-import all necessary modules
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from decouple import config
import streamlit as st
import warnings

###############################################################################################################################################################

# Suppress specific warnings
def suppress_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################################################################################################

# Set up API and database configurations
def setup_configurations():
    HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    db_path = "fancy_tshirts.db"  # Update this to the path of your SQLite3 database file
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    return HUGGINGFACEHUB_API_TOKEN, repo_id, db
###############################################################################################################################################################

# Initialize the Hugging Face LLM and SQL query chain
def initialize_llm_and_query_chain(HUGGINGFACEHUB_API_TOKEN, repo_id, db):
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
    generate_query = create_sql_query_chain(llm, db)
    return llm, generate_query

###############################################################################################################################################################

# Initialize the SQL database execution tool
def initialize_sql_execution_tool(db):
    execute_query = QuerySQLDataBaseTool(db=db)
    return execute_query

###############################################################################################################################################################

# Set up the answer prompt template
def setup_answer_prompt():
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )
    return answer_prompt

###############################################################################################################################################################

# Create and configure the overall chain
def setup_chain(generate_query, execute_query, rephrase_answer):
    chain = (
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
    )
    return chain

###############################################################################################################################################################

# Main function to invoke the entire process
def process_query(query_text):
    suppress_warnings()
    
    HUGGINGFACEHUB_API_TOKEN, repo_id, db = setup_configurations()
    llm, generate_query = initialize_llm_and_query_chain(HUGGINGFACEHUB_API_TOKEN, repo_id, db)
    execute_query = initialize_sql_execution_tool(db)
    answer_prompt = setup_answer_prompt()
    rephrase_answer = answer_prompt | llm | StrOutputParser()
    
    chain = setup_chain(generate_query, execute_query, rephrase_answer)
    
    # Invoke the chain with the user query
    res = chain.invoke({"question": query_text})
    
    return res

###############################################################################################################################################################

# Streamlit UI code
def streamlit_ui():
    st.title("🤖 **Ask Your Database!** ✨")

    # Text input for the user to enter a query
    query_input = st.text_input("Enter your query about T-Shirts:")

    # Submit button
    if st.button("Get Answer"):
        with st.spinner("Processing your query..."):
            result = process_query(query_input)
            st.success("Query processed successfully!")
        
        # Displaying the result
        st.subheader("Answer:")
        st.write(result)

###############################################################################################################################################################

if __name__ == "__main__":
    streamlit_ui()

###############################################################################################################################################################
