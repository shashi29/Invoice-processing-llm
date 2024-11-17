import streamlit as st
import json
import tempfile
from pathlib import Path
import os
from pdf2image import convert_from_path
import pytesseract
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

load_dotenv()
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
        Here is data: {data}'''
        
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

class InvoiceExtractorLLM:
    def __init__(self):
        self.llm_service_json = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=os.getenv('OPENAI_TEMPERATURE'),
            model_name=os.getenv('OPENAI_MODEL'),
            top_p=os.getenv('OPENAI_TOP_P'),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        self.prompt_structure_invoice = '''
        Extract all details from a given invoice in a structured format. Ensure that all information is correctly extracted, fixing any typos found. Do not make assumptions about the content, and make sure no details are missed. Only provide the extracted invoice data in the output.
        # Steps
        1. **Identify All Fields:** Carefully identify each field present in the invoice, including but not limited to invoice number, date, vendor details, product/service descriptions, quantities, prices, tax information, and total amount.
        2. **Correct Typos:** Fix any typos present in the invoice to ensure accurate information.
        3. **Avoid Assumptions:** Do not make assumptions or alter any content beyond correcting typographical errors.
        4. **Extract to Structured Format:** Extract the data and store it in a structured format to ensure readability and accessibility. Give a valid json.
        Also do not use nested json. Use as one one key-value format.
        # Notes
        - Standardize dates in the format YYYY-MM-DD and amounts with two decimal places. 
        - Be cautious of varying invoice formats, and ensure all available information is captured.
        '''
    
    def extract_text_from_image_pdf(self, pdf_path):
        images = convert_from_path(pdf_path)
        ocr_results = " "
        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img)
            ocr_results = ocr_results + text.strip()
        return ocr_results
    
    def structure_invoice_llm(self, pdf_path):
        ocr_results = self.extract_text_from_image_pdf(pdf_path)
        output = self.llm_service_json.invoke(self.prompt_structure_invoice + ocr_results)
        return ocr_results, output

class PDFViewer:
    def __init__(self):
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        if 'pdf_images' not in st.session_state:
            st.session_state.pdf_images = None
    
    def load_pdf(self, pdf_path):
        st.session_state.pdf_images = convert_from_path(pdf_path)
        st.session_state.total_pages = len(st.session_state.pdf_images)
    
    def next_page(self):
        if st.session_state.current_page < st.session_state.total_pages - 1:
            st.session_state.current_page += 1
    
    def prev_page(self):
        if st.session_state.current_page > 0:
            st.session_state.current_page -= 1
    
    def display_current_page(self):
        if st.session_state.pdf_images is not None:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.current_page == 0):
                    self.prev_page()
            
            with col2:
                st.markdown(f"<h4 style='text-align: center'>Page {st.session_state.current_page + 1} of {st.session_state.total_pages}</h4>", unsafe_allow_html=True)
            
            with col3:
                if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.current_page == st.session_state.total_pages - 1):
                    self.next_page()
            
            st.image(
                st.session_state.pdf_images[st.session_state.current_page],
                caption=f'Page {st.session_state.current_page + 1}',
                use_container_width=True
            )

def display_extracted_data(extracted_data):
    # Convert the extracted data to a DataFrame
    data = eval(extracted_data.content)
    df = pd.DataFrame([data]).T.reset_index()
    df.columns = ['Field', 'Value']
    
    # Style the DataFrame
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Field": st.column_config.TextColumn("Field", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="large")
        },
        use_container_width=True
    )

def main():
    st.set_page_config(
        page_title="Invoice Data Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem 2rem;
        }
        .stButton button {
            width: 100%;
        }
        .css-1v0mbdj {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Application header with improved styling
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5;'>
            📄 Invoice Data Extractor
        </h1>
        <p style='text-align: center; color: #666666; margin-bottom: 2rem;'>
            Upload your invoice PDF to extract structured data using AI processing
        </p>
    """, unsafe_allow_html=True)
    
    # File upload section with improved styling
    uploaded_file = st.file_uploader(
        "Choose an invoice PDF file",
        type=['pdf'],
        help="Upload a PDF invoice to begin extraction"
    )
    
    if uploaded_file is not None:
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📄 Invoice Preview", "📊 Extracted Data", "💬 Chat"])
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with tab1:
                pdf_viewer = PDFViewer()
                pdf_viewer.load_pdf(tmp_file_path)
                pdf_viewer.display_current_page()
            
            with tab2:
                with st.spinner('Extracting data from invoice...'):
                    # Initialize extractor and process invoice
                    invoice_service = InvoiceExtractorLLM()
                    ocr_result, extracted_data = invoice_service.structure_invoice_llm(tmp_file_path)
                    
                    # Display extracted data
                    display_extracted_data(extracted_data)
                    
                    # Add download options
                    col1, col2 = st.columns(2)
                    with col1:
                        json_str = json.dumps(eval(extracted_data.content), indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="extracted_invoice_data.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Convert to CSV
                        csv = pd.DataFrame([eval(extracted_data.content)]).T.to_csv()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name="extracted_invoice_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
            with tab3:
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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    # Add footer with instructions
    st.markdown("---")
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        1. Click the upload button or drag and drop a PDF invoice
        2. Switch between tabs to view the original invoice and extracted data
        3. Use the navigation buttons to move between pages
        4. Download the extracted data in JSON or CSV format
        
        ### Tips for Best Results
        - Ensure the PDF is clearly scanned
        - Check that text is legible
        - Make sure all important invoice details are visible
        """)

if __name__ == "__main__":
    main()