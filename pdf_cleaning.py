import re
import PyPDF2
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

def extract_raw_text_from_pdf(pdf_path: str, start_page: int=-1, last_page:int=0) -> List[dict]:
    """Extract text from PDF page by page"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_text = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                if page_num<start_page : pass
                if last_page!=0 and page_num>last_page : break
                print(page_num)
                text = page.extract_text()
                pages_text.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'lines': text.split('\n') if text else []
                })
            del pdf_reader
                
        logger.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text
    
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return []

def detect_headers_footers(pages_data: List[dict], threshold: float = 0.7) -> dict:
    """Detect repeated headers and footers across pages"""
    if len(pages_data) < 3:  # Need multiple pages to detect patterns
        return {"headers": set(), "footers": set()}
    
    # Collect first and last lines from each page
    first_lines = []
    last_lines = []
    
    for page in pages_data:
        lines = [line.strip() for line in page['lines'] if line.strip()]
        if lines:
            first_lines.append(lines[0])
            if len(lines) > 1:
                last_lines.append(lines[-1])
    
    # Find patterns that appear frequently
    def find_frequent_patterns(lines_list, min_frequency):
        pattern_count = {}
        for line in lines_list:
            # Normalize line (remove page numbers, etc.)
            normalized = re.sub(r'\b\d+\b', 'NUM', line)
            pattern_count[normalized] = pattern_count.get(normalized, 0) + 1
        
        frequent = set()
        for pattern, count in pattern_count.items():
            if count >= min_frequency:
                frequent.add(pattern)
        return frequent
    
    min_freq = max(2, int(len(pages_data) * threshold))
    
    headers = find_frequent_patterns(first_lines, min_freq)
    footers = find_frequent_patterns(last_lines, min_freq)
    
    logger.info(f"Detected {len(headers)} header patterns, {len(footers)} footer patterns")
    return {"headers": headers, "footers": footers}

def clean_page_numbers(text: str) -> str:
    """Remove various page number patterns"""
    patterns = [
        r'^\s*\d+\s*$',  # Standalone numbers
        r'^\s*-\s*\d+\s*-\s*$',  # -5-
        r'^\s*Page\s+\d+\s*$',  # Page 5
        r'^\s*\d+\s*/\s*\d+\s*$',  # 5/100
        r'^\s*\|\s*\d+\s*\|\s*$',  # |5|
        r'^\s*\d+\s*\|\s*.*$',  # 5 | Chapter Title
        r'^.*\|\s*\d+\s*$',  # Chapter Title | 5
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        is_page_number = False
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_page_number = True
                break
        
        if not is_page_number:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def remove_table_of_contents(text: str) -> str:
    """Remove table of contents sections"""
    toc_patterns = [
        r'table\s+of\s+contents.*?(?=chapter|section|\n\n[A-Z])',
        r'contents.*?(?=chapter|section|\n\n[A-Z])',
        r'index.*?(?=chapter|section|\n\n[A-Z])',
    ]
    
    for pattern in toc_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text

def remove_bibliography_references(text: str) -> str:
    """Remove bibliography and reference sections"""
    # Remove reference sections
    ref_patterns = [
        r'references.*?(?=chapter|section|\n\n[A-Z]|$)',
        r'bibliography.*?(?=chapter|section|\n\n[A-Z]|$)',
        r'works\s+cited.*?(?=chapter|section|\n\n[A-Z]|$)',
    ]
    
    for pattern in ref_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove inline citations like [1], (Smith, 2020), etc.
    citation_patterns = [
        r'\[\d+\]',  # [1], [23]
        r'\(\w+,?\s*\d{4}\)',  # (Smith, 2020)
        r'\(\w+\s+et\s+al\.,?\s*\d{4}\)',  # (Smith et al., 2020)
    ]
    
    for pattern in citation_patterns:
        text = re.sub(pattern, '', text)
    
    return text

def remove_urls_and_emails(text: str) -> str:
    """Remove URLs and email addresses"""
    # Remove URLs
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    text = re.sub(url_pattern, '', text)
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '', text)
    
    return text

def clean_formatting_artifacts(text: str) -> str:
    """Clean various PDF extraction artifacts"""
    # Fix hyphenated words split across lines
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Fix words split across lines (common PDF issue)
    text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    
    # Remove lines with mostly special characters (often formatting artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append('')
            continue
            
        # Skip lines that are mostly special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', line)) / len(line) if line else 0
        if special_char_ratio > 0.5 and len(line) < 50:
            continue
            
        # Skip lines that look like formatting (mostly dots, dashes, etc.)
        if re.match(r'^[\.\-_=\*\s]+$', line):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_clean_text_from_pdf(pdf_path: str, 
                               remove_headers_footers: bool = True,
                               remove_page_numbers: bool = True,
                               remove_toc: bool = True,
                               remove_references: bool = True,
                               remove_urls: bool = True) -> str:
    """
    Extract and clean text from PDF with multiple cleaning options
    
    Args:
        pdf_path: Path to PDF file
        remove_headers_footers: Remove repeated headers/footers
        remove_page_numbers: Remove page numbers
        remove_toc: Remove table of contents
        remove_references: Remove bibliography/references
        remove_urls: Remove URLs and emails
    
    Returns:
        Cleaned text string
    """
    
    # Extract raw text
    pages_data = extract_raw_text_from_pdf(pdf_path,16,150)
    if not pages_data:
        return ""
    
    # Detect headers and footers
    if remove_headers_footers:
        patterns = detect_headers_footers(pages_data)
        headers = patterns["headers"]
        footers = patterns["footers"]
    else:
        headers = set()
        footers = set()
    
    # Process each page
    cleaned_pages = []
    
    for page in pages_data:
        text = page['text']
        lines = page['lines']
        
        if remove_headers_footers:
            # Remove detected headers and footers
            cleaned_lines = []
            for line in lines:
                line_normalized = re.sub(r'\b\d+\b', 'NUM', line.strip())
                
                # Skip if matches header or footer pattern
                if line_normalized in headers or line_normalized in footers:
                    continue
                    
                cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
        
        cleaned_pages.append(text)
    
    # Combine all pages
    full_text = '\n'.join(cleaned_pages)
    
    # Apply various cleaning steps
    if remove_page_numbers:
        full_text = clean_page_numbers(full_text)
    
    if remove_toc:
        full_text = remove_table_of_contents(full_text)
    
    if remove_references:
        full_text = remove_bibliography_references(full_text)
    
    if remove_urls:
        full_text = remove_urls_and_emails(full_text)
    
    # Always clean formatting artifacts
    full_text = clean_formatting_artifacts(full_text)
    
    # Final cleanup
    full_text = '\n'.join(line for line in full_text.split('\n') if line.strip())
    
    logger.info(f"Cleaned text: {len(full_text)} characters")
    return full_text


data_dir = Path.cwd() / "data"
def get_pdf_files_pathlib(directory_path=str(data_dir)):
    directory = Path(directory_path)
    
    return [
        {
            'filename': pdf_file.name,
            'path': str(pdf_file)
        }
        for pdf_file in directory.rglob('*.pdf')
    ]
# def main():
#     files = get_pdf_files_pathlib()
#     pdf_file = files[0]['path']
    
#     if not Path(pdf_file).exists():
#         print(f"❌ PDF file '{pdf_file}' not found")
#         return
#     cleaned_text = extract_clean_text_from_pdf(pdf_file)
    
#     # Save cleaned text to file
#     output_file = "cleaned_text.txt"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(cleaned_text)
    
#     print(f"\n✅ Cleaned text saved to: {output_file}")

# if __name__ == "__main__":
#     main() 