from pdf2image import convert_from_path
import fitz  # PyMuPDF

def pdf_to_images(pdf_path, dpi=300):
    poppler_path = r"C:\poppler-23.05.0\poppler-24.08.0\Library\bin"
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return pages


def extract_text_blocks(pdf_path):
    """Extract text blocks with bbox and font size using PyMuPDF."""
    doc = fitz.open(pdf_path)
    all_pages_blocks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_blocks = []
        for b in blocks:
            if b['type'] == 0:  # text block
                for line in b["lines"]:
                    line_text = ""
                    bbox = None
                    font_size = None
                    for span in line["spans"]:
                        line_text += span["text"]
                        bbox = span["bbox"]  # (x0, y0, x1, y1)
                        font_size = span["size"]
                    page_blocks.append({
                        "text": line_text.strip(),
                        "bbox": bbox,
                        "font_size": font_size
                    })
        all_pages_blocks.append(page_blocks)
    return all_pages_blocks
