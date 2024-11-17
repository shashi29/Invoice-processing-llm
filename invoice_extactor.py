import os
import pytesseract
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class InvoiceExtractorLLM():
    def __init__(self):
        self.llm_service_json =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                            temperature=os.getenv('OPENAI_TEMPERATURE'),
                            model_name=os.getenv('OPENAI_MODEL'),
                            top_p=os.getenv('OPENAI_TOP_P'),
                            model_kwargs={ "response_format": { "type": "json_object" } })
        self.llm_service =  ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                            temperature=os.getenv('OPENAI_TEMPERATURE'),
                            model_name=os.getenv('OPENAI_MODEL'),
                            top_p=os.getenv('OPENAI_TOP_P'))
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
        return output
    
    
if __name__ == "__main__":
    import pandas as pd
    pdf_path = "/root/PetroChoice - Saia - 10700904540 (1).pdf"
    invoice_service = InvoiceExtractorLLM()
    out = invoice_service.structure_invoice_llm(pdf_path)
    data = eval(out.content)
    # Flatten the main dictionary
    flat_data = {k: v for k, v in data.items()}

    # Convert main data to a single-row DataFrame
    main_df = pd.DataFrame([flat_data]).T
    
    print(main_df)
    