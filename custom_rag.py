import json
import os
import uuid
import zlib
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, UpdateStatus, OptimizersConfigDiff
from tqdm import tqdm  # Import tqdm for progress tracking
from typing import List, Dict, Any

import json
from pathlib import Path
from collections import OrderedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#pandas ai dependency
import pandas as pd
from pandasai import SmartDatalake
from pandasai import Agent, SmartDataframe

from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI, OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoustomRag:
    def __init__(self, embedding_model_name: str, quadrant_url: str, collection_name: str, limit: int=5):
        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        self.qdrant_client = QdrantClient(base_url=quadrant_url)
        self.collection_name = collection_name
        self.limit = limit
        self.llm_service =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'))
        
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

        logger.info(f"Search completed for query: {question} with {len(results)} results.")
        return final_answer #{"final_answer": final_answer, "metadata": metadata}
    
    def get_answer_from_all_context(self, question):
        query = self.embedding_model.encode(question).tolist()
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=40
        )
        
        results = [{"score": hit.score, "payload": hit.payload} for hit in search_results]
        for text_info in results:
            texts = text_info["payload"]["text"]
            formatted_prompt = self._get_formatted_prompt(texts, question)
            final_answer = self.llm_service.invoke(formatted_prompt)
            print("******************************************************************************\n")
            print(final_answer)
            if "Not able to find relevant context" not in final_answer.content :
                return final_answer.content
        
        return  "Not able to find relevant context"
    
    def _generate_final_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        texts = [result["payload"]["text"] for result in results]
        combined_text = "\n".join(texts)
        formatted_prompt = self._get_formatted_prompt(combined_text, query)
        
        try:
            final_answer = self.llm_service.invoke(formatted_prompt)
            return final_answer
        except Exception as e:
            logger.error(f"Error generating final answer with LLM: {e}", exc_info=True)
            return "Failed to generate final answer."

    @staticmethod
    def _get_formatted_prompt(combined_text: str, query: str) -> str:
        return f"""
        Given the following information: {combined_text}
        Please answer this question based solely on the information provided above: {query}
        Remember to use only the information from the given text in your answer. 
        Do not introduce any external information or make assumptions beyond what is explicitly stated in the text.
        If you can not answer the question then return "Not able to find relvant context"
        """
        
class CSVQuestionAnswer():
    def __init__(self, excel_path):
        self.dataframes_list = self.create_dataframe_from_excel(excel_path)
        self.markdown_data = '\n'.join([df.head(5).to_markdown() for df in self.dataframes_list])
        self.llm_service =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'))
        self.agent = Agent(self.dataframes_list, config={"llm": self.llm_service, "verbose": False, "enable_cache": False, "max_retries": 3})
                
                        
    def create_dataframe_from_excel(self, excel_path):
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        dataframes_list = list()
        # Access each sheet as a separate DataFrame
        for sheet_name, df in excel_data.items():
            print(f"Sheet name: {sheet_name}")
            dataframes_list.append(df)
        return dataframes_list

    def get_planner_instruction_with_data(self, input_prompt, data):
        return f''' 
        Enhance the instructions for using a Pandas DataFrame without including specific code.
        Exclude steps related to importing libraries and loading data. 
        User input prompt: {input_prompt} 
        Here is sample data for column reference: {data}'''
        
    def get_answer(self, question):
        updated_instruction = self.get_planner_instruction_with_data(question, self.markdown_data)
        updated_instruction = self.llm_service.invoke(updated_instruction)
        rephrased_query = self.agent.rephrase_query(updated_instruction.content)
        response_content = self.agent.chat(rephrased_query)
        return response_content

        
def get_final_answer(question, text_rag, csv_rag):
    llm_service =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'))
    
    text_answer = text_rag.get_answer(question)
    csv_answer = csv_rag.get_answer(question)
    final_prompt = f"""
    Question: {question}

    Text-based Answer:{text_answer}
    CSV-based Answer: {csv_answer}

    Synthesize a final, comprehensive answer using both the provided text-based answer and CSV data.

    To complete this task, integrate and reconcile both sources of information—combining the textual insights with the empirical CSV-based findings.

    # Steps

    1. **Understand the Question**: Clearly comprehend the original question to ensure that both sources of information are being directly applied to answer it.
    2. **Analyze Both Data Sources**:
    - **Text-based Answer**: Extract core insights and arguments from the text.
    - **CSV-based Answer**: Extract relevant quantitative or data-oriented insights from the CSV information.
    3. **Synthesize Information**:
    - Combine qualitative insights from the text-based answer with the empirical support from the CSV-based answer.
    - Resolve any conflicting information between the two sources.
    - Ensure conciseness, clarity, and a comprehensive response to the question using the strengths of each source.
    
    # Output Format

    - The response should be a paragraph that integrates the qualitative and quantitative data cohesively.
    - The response should address the specific question comprehensively and directly.
    - Length: Approximately 3-5 sentences, more if needed to achieve coherence.
    - Structured logically, starting by addressing significant points from both answers, and ensuring that the final synthesis is clearly presented.

    # Notes

    - Prioritize synthesizing a cohesive answer that makes logical sense from the two sources.
    - If you encounter any conflicting information, determine which source holds more authority based on the type of question and point that out as needed.
    - Aim for clarity, ensuring the reader can understand how the text and CSV data complement each other.    
    """
    final_answer = llm_service.invoke(final_prompt)
    return final_answer.content
        
if __name__ == "__main__":
    text_rag = CoustomRag(
        embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
        quadrant_url=os.getenv("QDRANT_URL"),
        collection_name="contract_collection",
        limit=5
    )
    
    #text_out = text_rag.get_answer("""Are the ship date (05-18-23), delivery date (05-22-23), and reference date (05-22-23) consistent with the agreed timelines within the contract?""")
    #print(text_out)
    # question = "This agreement is made between which two parties?"
    # out = text_rag.get_answer_from_all_context(question)
    # print(out)
    
    csv_rag = CSVQuestionAnswer("/root/code/Invoice-processing-llm/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)_table.xlsx")
    # csv_out = csv_rag.get_answer('What is the discount from FL to FL')
    # print(csv_out)
    
    
    # ###########Now we need to a way which takes answer from both of these give to llm service and then give the final answer
    # #Whe common question is being asked to someone
    question = """What is the discount from FL to FL ?"""
    final = get_final_answer(question, text_rag, csv_rag)
    print(final)