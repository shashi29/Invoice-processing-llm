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
        final_answer = self._generate_final_answer(results, query)
        #final_answer = ""
        metadata = [result["payload"] for result in results]

        logger.info(f"Search completed for query: {query} with {len(results)} results.")
        return final_answer #{"final_answer": final_answer, "metadata": metadata}
    
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
        """
        
class CSVQuestionAnswer():
    def __init__(self, excel_path):
        self.dataframes_list = self.create_dataframe_from_excel(excel_path)
        self.markdown_data = '\n'.join([df.head(5).to_markdown() for df in self.dataframes_list])
        self.llm_service =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'))
        self.agent = Agent(self.dataframes_list, config={"llm": self.llm_service, "verbose": False, "enable_cache": False, "max_retries": 10})
                
                        
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
        Here is data: {data}'''
        
    def get_answer(self, question):
        input_text = "What is the Discount between AL and TX and what is the minium value?"
        updated_instruction = self.get_planner_instruction_with_data(input_text, self.markdown_data)
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

    Based on the information provided in the text and the CSV data, please synthesize a final, comprehensive answer to the original question.
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
    
    # text_out = text_rag.get_answer("What are the different compliance conditions that we talk about here")
    # print(text_out)
    
    csv_rag = CSVQuestionAnswer("/root/code/Invoice-processing-llm/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1)_table.xlsx")
    # csv_out = csv_rag.get_answer('What is discount between AR and LA')
    # print(csv_out)
    
    
    ###########Now we need to a way which takes answer from both of these give to llm service and then give the final answer
    #Whe common question is being asked to someone
    question = "What is discount between AR and LA"
    final = get_final_answer(question, text_rag, csv_rag)
    print(final)