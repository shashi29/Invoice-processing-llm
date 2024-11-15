import streamlit as st
import pdfplumber
import io
from PIL import Image

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF document."""
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                # Convert table to formatted string
                table_text = '\n'.join([' | '.join([str(cell) if cell else '' for cell in row]) for row in table])
                tables_text.append(table_text)
    return '\n\n'.join(tables_text)

def main():
    st.set_page_config(layout="wide")
    st.title("PDF Table Viewer and Extractor")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create columns for layout
        left_col, right_col = st.columns(2)

        # Read PDF file
        pdf_bytes = io.BytesIO(uploaded_file.read())

        # Get total number of pages
        with pdfplumber.open(pdf_bytes) as pdf:
            total_pages = len(pdf.pages)

        # Initialize session state for page number
        if 'page_num' not in st.session_state:
            st.session_state.page_num = 0

        with left_col:
            st.subheader("PDF Viewer")

            # Navigation controls
            col1, col2, col3, col4 = st.columns([1, 3, 3, 1])

            with col1:
                if st.button("←", key="prev"):
                    if st.session_state.page_num > 0:
                        st.session_state.page_num -= 1

            with col2:
                st.write(f"Page {st.session_state.page_num + 1} of {total_pages}")

            with col4:
                if st.button("→", key="next"):
                    if st.session_state.page_num < total_pages - 1:
                        st.session_state.page_num += 1

            # Display PDF page
            with pdfplumber.open(pdf_bytes) as pdf:
                page = pdf.pages[st.session_state.page_num]
                img = page.to_image()
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                st.image(img_byte_arr, use_column_width=True)

        with right_col:
            st.subheader("Extracted Tables")

            # Extract and display tables from the PDF
            tables_text = extract_tables_from_pdf(pdf_bytes)
            if tables_text:
                st.text_area("Table Content", tables_text, height=500)
            else:
                st.write("No tables found in the PDF.")

if __name__ == "__main__":
    main()