import re

import pdfplumber
from PIL import Image
from qwen_vl_utils import process_vision_info


def get_txt_from_doc(image, prompt, model, processor, device):
    """Extracts text from an image (one page of a PDF) based on a user prompt using a specified model.

    Args:
        image: A single image extracted from a PDF.
        prompt: A user-provided prompt to guide the model's text extraction.
        model: The model used for text extraction (e.g., "Qwen/Qwen2.5-VL-3B-Instruct").
        processor: The processor associated with the model for handling inputs and outputs.
        device: GPU or CPU

    Returns:
        A list of extracted and cleaned text strings.
    """
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
    """Converts a PDF file into a list of images, resizing them to the specified dimensions.

    Args:
        pdf_path: Path to the PDF file.
        target_size: A tuple (width, height) specifying the target size of the images.
        resolution: The resolution (DPI) for rendering the PDF pages.

    Returns:
        A list of PIL.Image objects representing the pages of the PDF.
    """
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            img = page.to_image(resolution=resolution).original
            img = img.convert("RGB")  # Convert to RGB
            img = img.resize(target_size, Image.LANCZOS)  # Resize the image
            images.append(img)
    return images


def clean_text(text):
    """Cleans and formats raw text by removing unnecessary characters, lines, and spaces.

    Args:
        text: The raw text to be cleaned.

    Returns:
        The cleaned and formatted text.
    """
    # Remove bold text and headings (asterisks and hashes)
    text = re.sub(r"[*#]", "", text)

    # Remove separator lines (e.g., "---")
    text = re.sub(r"^-+$", "", text, flags=re.MULTILINE)

    # Remove leading and trailing whitespace
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

    # Remove extra blank lines
    text = re.sub(r"\n+", "\n", text).strip()

    return text
