import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from pdf2image import convert_from_path
import pytesseract
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime

# Load environment variables
load_dotenv()

@dataclass
class InvoiceData:
    """Data class to store structured invoice information"""
    invoice_number: str
    date: datetime
    vendor_details: Dict[str, str]
    line_items: list
    totals: Dict[str, float]
    additional_info: Dict[str, Any]

class InvoiceProcessor:
    """Handles all invoice-related processing operations"""
    
    def __init__(self):
        self.llm_service = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
            model_name=os.getenv('OPENAI_MODEL', 'gpt-4'),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        self.qa_llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
            model_name=os.getenv('OPENAI_MODEL', 'gpt-4')
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR"""
        try:
            images = convert_from_path(pdf_path)
            ocr_results = []
            
            for img in images:
                text = pytesseract.image_to_string(img)
                ocr_results.append(text.strip())
            
            return "\n".join(ocr_results)
        except Exception as e:
            st.error(f"Error during OCR processing: {str(e)}")
            raise

    def structure_invoice_data(self, ocr_text: str) -> Dict[str, Any]:
        """Extract structured data from OCR text using chain-of-thought reasoning"""
        prompt = """
        Process this invoice text using step-by-step reasoning to extract accurate information:

        1. First, identify key invoice metadata:
           - Invoice number
           - Date
           - Payment terms
           - Due date

        2. Then, extract vendor information:
           - Company name
           - Address
           - Contact details
           - Tax ID/Registration numbers

        3. Next, analyze line items:
           - Product/service descriptions
           - Quantities
           - Unit prices
           - Line item totals

        4. Finally, extract financial totals:
           - Subtotal
           - Tax amounts (itemized by type)
           - Discounts
           - Total amount due

        Format the output as a JSON object with the following structure:
        {
            "metadata": {
                "invoice_number": "",
                "date": "",
                "payment_terms": "",
                "due_date": ""
            },
            "vendor": {
                "name": "",
                "address": "",
                "contact": {},
                "tax_info": {}
            },
            "line_items": [
                {
                    "description": "",
                    "quantity": 0,
                    "unit_price": 0.0,
                    "total": 0.0
                }
            ],
            "totals": {
                "subtotal": 0.0,
                "tax": {},
                "discounts": 0.0,
                "total": 0.0
            },
            "additional_info": {}
        }

        Invoice text to process:
        {text}
        """
        
        try:
            response = self.llm_service.invoke(prompt.format(text=ocr_text))
            return json.loads(response.content)
        except Exception as e:
            st.error(f"Error processing invoice data: {str(e)}")
            raise

    def answer_question(self, question: str, invoice_data: Dict[str, Any]) -> str:
        """Generate answers to questions about the invoice using chain-of-thought reasoning"""
        prompt = f"""
        Using the following invoice data, answer this question: "{question}"

        Think through this step-by-step:
        1. Identify which parts of the invoice data are relevant to the question
        2. Extract the relevant information
        3. Formulate a clear, concise answer
        4. Verify the answer against the original data
        
        Invoice data:
        {json.dumps(invoice_data, indent=2)}
        """
        
        try:
            response = self.qa_llm.invoke(prompt)
            return response.content
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            raise

class InvoiceUI:
    """Handles all Streamlit UI components"""
    
    def __init__(self):
        self.processor = InvoiceProcessor()
        self.setup_session_state()

    @staticmethod
    def setup_session_state():
        """Initialize session state variables"""
        if 'invoice_data' not in st.session_state:
            st.session_state.invoice_data = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def display_pdf(self, pdf_file: str):
        """Display PDF in the Streamlit UI"""
        with st.container():
            images = convert_from_path(pdf_file)
            for idx, image in enumerate(images):
                st.image(image, caption=f'Page {idx + 1}', use_column_width=True)

    def display_structured_data(self, data: Dict[str, Any]):
        """Display structured invoice data in an organized format"""
        with st.expander("📊 Structured Invoice Data", expanded=True):
            # Display metadata
            st.subheader("Invoice Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Invoice Number:", data['metadata']['invoice_number'])
                st.write("Date:", data['metadata']['date'])
            with col2:
                st.write("Payment Terms:", data['metadata']['payment_terms'])
                st.write("Due Date:", data['metadata']['due_date'])

            # Display vendor information
            st.subheader("Vendor Information")
            st.json(data['vendor'])

            # Display line items in a table
            st.subheader("Line Items")
            if data['line_items']:
                st.table(data['line_items'])

            # Display totals
            st.subheader("Totals")
            st.json(data['totals'])

    def run(self):
        """Main UI execution"""
        st.set_page_config(page_title="Invoice Q&A System", page_icon="🧾", layout="wide")
        st.title("🧾 Interactive Invoice Analysis System")

        # File upload section
        uploaded_file = st.file_uploader("Upload an invoice (PDF)", type=['pdf'])

        if uploaded_file:
            # Process the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Create two columns for PDF and data
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📄 Original Invoice")
                    self.display_pdf(tmp_file_path)

                with col2:
                    # Process invoice if not already processed
                    if st.session_state.invoice_data is None:
                        with st.spinner("Processing invoice..."):
                            ocr_text = self.processor.extract_text_from_pdf(tmp_file_path)
                            st.session_state.invoice_data = self.processor.structure_invoice_data(ocr_text)

                    # Display structured data
                    self.display_structured_data(st.session_state.invoice_data)

                # Q&A section
                st.markdown("---")
                st.subheader("💬 Ask Questions About This Invoice")
                
                # Question input
                question = st.text_input("Ask a question about the invoice:")
                if question:
                    with st.spinner("Generating answer..."):
                        answer = self.processor.answer_question(question, st.session_state.invoice_data)
                        st.session_state.chat_history.append({"question": question, "answer": answer})

                # Display chat history
                if st.session_state.chat_history:
                    st.subheader("Previous Questions and Answers")
                    for qa in st.session_state.chat_history:
                        with st.expander(f"Q: {qa['question']}", expanded=True):
                            st.write("A:", qa['answer'])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Cleanup
                os.unlink(tmp_file_path)

        # Instructions
        with st.sidebar:
            st.subheader("📋 Instructions")
            st.markdown("""
            1. Upload a PDF invoice using the file uploader
            2. Wait for the processing to complete
            3. Review the structured data extracted from your invoice
            4. Ask questions about the invoice in the chat section
            5. View previous questions and answers in the history
            
            **Tips for best results:**
            - Ensure PDF is clearly scanned
            - Ask specific questions about invoice details
            - Use clear, concise questions
            """)

if __name__ == "__main__":
    app = InvoiceUI()
    app.run()