import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
import pandas as pd

# Load environment variables
load_dotenv()


#pandas ai dependency
import pandas as pd
from pandasai import SmartDatalake
from pandasai import Agent, SmartDataframe

from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI, OpenAI
class CoustomRag:
    def __init__(self, embedding_model_name: str, quadrant_url: str, collection_name: str, limit: int=5):
        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        self.qdrant_client = QdrantClient(base_url=quadrant_url)
        self.collection_name = collection_name
        self.limit = limit
        self.llm_service = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE')),
            model_name=os.getenv('OPENAI_MODEL'),
            top_p=float(os.getenv('OPENAI_TOP_P'))
        )

    def get_answer(self, question: str):
        query = self.embedding_model.encode(question).tolist()
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=self.limit
        )
        
        results = [{"score": hit.score, "payload": hit.payload} for hit in search_results]
        # Process the results to include final answer and metadata
        final_answer = self._generate_final_answer(results, question)
        #final_answer = ""
        metadata = [result["payload"] for result in results]

        # logger.info(f"Search completed for query: {question} with {len(results)} results.")
        return final_answer #{"final_answer": final_answer, "metadata": metadata}
    
    def get_answer_from_all_context(self, question):
        query = self.embedding_model.encode(question).tolist()
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=5
        )
        
        results = [{"score": hit.score, "payload": hit.payload} for hit in search_results]
        for text_info in results:
            texts = text_info["payload"]["text"]
            formatted_prompt = self._get_formatted_prompt(texts, question)
            final_answer = self.llm_service.invoke(formatted_prompt)
            if "Not able to find relevant context" not in final_answer.content:
                return final_answer.content
        
        return "Not able to find relevant context"

    @staticmethod
    def _get_formatted_prompt(combined_text: str, query: str) -> str:
        return f"""
        Given the following information: {combined_text}
        Please answer this question based solely on the information provided above: {query}
        Remember to use only the information from the given text in your answer. 
        Do not introduce any external information or make assumptions beyond what is explicitly stated in the text.
        If you can not answer the question then return "Not able to find relevant context"
        """

class CSVQuestionAnswer:
    def __init__(self, excel_path):
        self.dataframes_list = self.create_dataframe_from_excel(excel_path)
        self.markdown_data = '\n'.join([df.head(5).to_markdown() for df in self.dataframes_list])
        self.llm_service = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE')),
            model_name=os.getenv('OPENAI_MODEL'),
            top_p=float(os.getenv('OPENAI_TOP_P'))
        )
        self.agent = Agent(self.dataframes_list, config={
            "llm": self.llm_service,
            "verbose": False,
            "enable_cache": False,
            "max_retries": 3
        })

    def create_dataframe_from_excel(self, excel_path):
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        return list(excel_data.values())

    def get_planner_instruction_with_data(self, input_prompt, data):
        return f''' 
        Enhance the instructions for using a Pandas DataFrame without including specific code.
        Exclude steps related to importing libraries and loading data. 
        User input prompt: {input_prompt} 
        Here is data: {data}'''

    def get_answer(self, question):
        updated_instruction = self.get_planner_instruction_with_data(question, self.markdown_data)
        updated_instruction = self.llm_service.invoke(updated_instruction)
        rephrased_query = self.agent.rephrase_query(updated_instruction.content)
        return self.agent.chat(rephrased_query)

def get_final_answer(question, text_rag, csv_rag):
    llm_service = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=float(os.getenv('OPENAI_TEMPERATURE')),
        model_name=os.getenv('OPENAI_MODEL'),
        top_p=float(os.getenv('OPENAI_TOP_P'))
    )
    
    text_answer = text_rag.get_answer_from_all_context(question)
    csv_answer = csv_rag.get_answer(question)
    
    final_prompt = f"""
    Question: {question}

    Text-based Answer: {text_answer}
    CSV-based Answer: {csv_answer}

    Synthesize a final, comprehensive answer using both the provided text-based answer and CSV data.
    Integrate and reconcile both sources of information, combining textual insights with empirical CSV-based findings.
    """
    
    final_answer = llm_service.invoke(final_prompt)
    return final_answer.content

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Contract Q&A System",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Contract and Data Q&A System")
    st.markdown("Ask questions about your contracts and related data")

    # Initialize RAG systems
    if 'text_rag' not in st.session_state:
        st.session_state.text_rag = CoustomRag(
            embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
            quadrant_url=os.getenv("QDRANT_URL"),
            collection_name="contract_collection",
            limit=5
        )

    if 'csv_rag' not in st.session_state:
        st.session_state.csv_rag = CSVQuestionAnswer("/root/code/Invoice-processing-llm/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)_table.xlsx")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about the contract?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_final_answer(
                    prompt,
                    st.session_state.text_rag,
                    st.session_state.csv_rag
                )
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()