import pdfplumber
import pandas as pd
import json
from pathlib import Path
import logging


class PDFExtractor:
    def __init__(self, pdf_path):
        """
        Initialize the PDF extractor with a path to the PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = self.pdf_path.stem  # Folder name will be the PDF name without extension
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self):
        """
        Extract all text from the PDF document.
        
        Returns:
            dict: Dictionary with page numbers as keys and extracted text as values
        """
        text_content = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_content[f"page_{page_num}"] = text.strip()
                    else:
                        self.logger.warning(f"No text found on page {page_num}")
            
            return text_content
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def extract_tables(self):
        """
        Extract all tables from the PDF document.
        
        Returns:
            dict: Dictionary with table dataframes categorized by page and table number
        """
        table_content = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    if tables:
                        # Convert each table to a pandas DataFrame
                        processed_tables = []
                        for table in tables:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            processed_tables.append(df)
                        
                        table_content[f"page_{page_num}"] = processed_tables
                        self.logger.info(f"Found {len(tables)} tables on page {page_num}")
                    
            return table_content
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
            raise
    
    def save_to_files(self):
        """
        Save extracted content to files within a directory named after the PDF.
        """
        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save all text content into a JSON file
            text_content = self.extract_text()
            json_file_path = output_path / f"{self.pdf_path.stem}_text.json"
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(text_content, json_file, indent=4, ensure_ascii=False)
            self.logger.info(f"Text content saved to {json_file_path}")
            
            # Save all tables into a single Excel file
            table_content = self.extract_tables()
            if table_content:
                excel_file_path = output_path / f"{self.pdf_path.stem}_tables.xlsx"
                with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
                    for page_num, tables in table_content.items():
                        for table_num, table in enumerate(tables, 1):
                            sheet_name = f"Page{page_num}_Table{table_num}"
                            table.to_excel(writer, index=False, sheet_name=sheet_name)
                
                self.logger.info(f"Table content saved to {excel_file_path}")
            else:
                self.logger.info("No tables found to save.")
            
        except Exception as e:
            self.logger.error(f"Error saving files: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    fpdf_path = "/root/PetroChoice_SAIA_LTL_Exhibit E_ 08.22.2022 (1) (1).pdf"
    extractor = PDFExtractor(fpdf_path)
    
    # Save all extracted content
    extractor.save_to_files()
