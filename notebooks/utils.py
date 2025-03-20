from qwen_vl_utils import process_vision_info
from PIL import Image

import pdfplumber
import re

def smart_chunking(paragraphs_text, tokenizer):
    """
    Splits paragraphs into semantically meaningful chunks with size constraints.
    
    Args:
        paragraphs_text (list[str]): List of raw paragraph texts to process
        
    Returns:
        list[str]: List of text chunks meeting the requirements:
            - Max chunk size: 1024 tokens
            - Overlap between chunks: 150 tokens
            - Preserves paragraph boundaries where possible
    """
    MAX_CHUNK_SIZE = 1024
    OVERLAP_SIZE = round(MAX_CHUNK_SIZE * 0.15)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para_text in paragraphs_text:
        para_tokens = tokenizer.encode(para_text, add_special_tokens=False)
        para_length = len(para_tokens)
        
        if para_length < 5:
            continue
            
        if para_length > MAX_CHUNK_SIZE:
            sub_paras = split_large_paragraph(para_text, MAX_CHUNK_SIZE, OVERLAP_SIZE, tokenizer)
            for sub_para in sub_paras:
                chunks.extend(smart_chunking([sub_para], tokenizer))
            continue
            
        if current_length + para_length > MAX_CHUNK_SIZE:
            chunks.append(merge_paragraphs(current_chunk))
            
            overlap_tokens = []
            while current_chunk and (len(overlap_tokens) < OVERLAP_SIZE):
                overlap_tokens = tokenizer.encode(current_chunk[-1], add_special_tokens=False) + overlap_tokens
                current_chunk.pop()
                
            current_chunk = [tokenizer.decode(overlap_tokens[-OVERLAP_SIZE:])] if overlap_tokens else []
            current_length = len(overlap_tokens)
            
        current_chunk.append(para_text)
        current_length += para_length
        
    if current_chunk:
        chunks.append(merge_paragraphs(current_chunk))
        
    return chunks

def split_large_paragraph(text, max_size, overlap, tokenizer):
    """
    Splits a single large paragraph into smaller chunks with overlap
    
    Args:
        text (str): Input text to split
        max_size (int): Maximum tokens per chunk (must be > overlap)
        overlap (int): Number of overlapping tokens between chunks (must be < max_size)
        tokenizer: Tokenizer instance
        
    Returns:
        list[str]: List of sub-paragraphs with overlapping regions
    
    Raises:
        ValueError: If overlap >= max_size
    """
    # Validate input parameters
    if overlap >= max_size:
        raise ValueError(f"Overlap ({overlap}) must be smaller than max_size ({max_size})")
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    sub_paras = []
    
    start = 0
    safety_counter = 0
    max_iterations = len(tokens) * 2  # Fallback to prevent infinite loops
    
    while start < len(tokens) and safety_counter < max_iterations:
        end = min(start + max_size, len(tokens))
        sub_para = tokenizer.decode(tokens[start:end])
        sub_paras.append(sub_para)
        
        # Calculate new start position with overlap
        new_start = end - overlap
        
        # Prevent infinite loops by ensuring forward progress
        if new_start <= start:
            new_start = start + 1  # Force progress by at least 1 token
            
        start = new_start
        safety_counter += 1

    # Final fallback truncation if still stuck
    if safety_counter >= max_iterations:
        sub_paras.append(tokenizer.decode(tokens[:max_size]))
        
    return sub_paras

def merge_paragraphs(paragraphs):
    """
    Merges multiple paragraphs into a single text chunk
    
    Args:
        paragraphs (list[str]): List of paragraph texts
        
    Returns:
        str: Combined text with paragraphs separated by newlines
    """
    return "\n\n".join(paragraphs)

def get_txt_from_doc(image, prompt, model, processor, device):
    '''Extracts text from an image (one page of a PDF) based on a user prompt using a specified model.

    Args:
        image: A single image extracted from a PDF.
        prompt: A user-provided prompt to guide the model's text extraction.
        model: The model used for text extraction (e.g., "Qwen/Qwen2.5-VL-3B-Instruct").
        processor: The processor associated with the model for handling inputs and outputs.
        device: GPU or CPU

    Returns:
        A list of extracted and cleaned text strings.
    '''
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Remove extra spaces
    output_text = [text.strip() for text in output_text]  

    return output_text


# util.py
def pdf_to_images(pdf_path, target_size=(760, 1024), resolution=225):
    '''Converts a PDF file into a list of images, resizing them to the specified dimensions.

    Args:
        pdf_path: Path to the PDF file.
        target_size: A tuple (width, height) specifying the target size of the images.
        resolution: The resolution (DPI) for rendering the PDF pages.

    Returns:
        A list of PIL.Image objects representing the pages of the PDF.
    '''
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=resolution).original
            img = img.convert("RGB")  # Convert to RGB
            img = img.resize(target_size, Image.LANCZOS)  # Resize the image
            images.append(img)
    return images


def clean_text(text):
    '''Cleans and formats raw text by removing unnecessary characters, lines, and spaces.

    Args:
        text: The raw text to be cleaned.

    Returns:
        The cleaned and formatted text.
    '''
    # Remove bold text and headings (asterisks and hashes)
    text = re.sub(r'[*#]', '', text)

    # Remove separator lines (e.g., "---")
    text = re.sub(r'^-+$', '', text, flags=re.MULTILINE)

    # Remove leading and trailing whitespace
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)

    # Remove extra blank lines
    text = re.sub(r'\n+', '\n', text).strip()

    return text