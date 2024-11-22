import os
import fitz  # PyMuPDF for PDF processing
import pdfplumber  # For table extraction from PDFs
from PIL import Image, ImageChops  # Image processing
import numpy as np
import io


# Function to check if an image is a single color (solid color)
def is_single_color(image):
    """
    Checks if the provided image is of a single color.

    Args:
        image (PIL.Image): The image to check.

    Returns:
        bool: True if the image is a single color, False otherwise.
    """
    img_array = np.array(image)
    return np.all(img_array == img_array[0, 0])


# Function to verify if two images are identical
def are_images_equal(img1, img2):
    """
    Checks if two images are identical by comparing their histograms.

    Args:
        img1 (PIL.Image): The first image to compare.
        img2 (PIL.Image): The second image to compare.

    Returns:
        bool: True if images are identical, False otherwise.
    """
    return img1.histogram() == img2.histogram()


# Function to check if a PDF contains tables
def has_tables(pdf_path):
    """
    Determines if a PDF file contains tables.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        bool: True if tables are found, False otherwise.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:  # Checks if tables exist on the page
                return True
    return False


# Main function to process a PDF file and save a modified version
def clean_pdf(pdf_path):
    """
    Processes a PDF by applying several filters to images:
    - Removes fully black, single-color, and reference duplicate images.
    - Optionally checks if the PDF contains tables.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple: Path to the modified PDF and a boolean flag indicating if tables are present.
    """

    # Directory containing reference images to compare for duplication
    reference_directory = 'reference'
    reference_images = []  # List to store reference images

    # Load all reference images for comparison
    for filename in os.listdir(reference_directory):
        if filename.endswith('.png'):
            img_path = os.path.join(reference_directory, filename)
            image = Image.open(img_path).convert("RGB")
            reference_images.append(image)

    # Set the path for the modified PDF file
    output_pdf_path = f"{os.path.splitext(pdf_path)[0]}_modified.pdf"

    # Check if the PDF contains tables
    contains_tables = has_tables(pdf_path)

    # Open the PDF using PyMuPDF (fitz)
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        # Iterate over each image in the current page
        for image in image_list:
            image_id = image[7]  # Image identifier on the page
            xref = image[0]  # Cross-reference for the image

            # Extract image bytes from the PDF
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Load the extracted image as a PIL image
            extracted_image = Image.open(io.BytesIO(image_bytes))

            # Filter 1: Check if the image is completely black
            black_image = Image.new("RGB", extracted_image.size, (0, 0, 0))
            if ImageChops.difference(extracted_image.convert("RGB"), black_image).getbbox() is None:
                print(f"[!] Page {page_num + 1}, Image {image_id} is completely black and will be discarded.")
                continue  # Skip to the next image

            # Filter 2: Check if the image is a single color
            if is_single_color(extracted_image.convert("RGB")):
                print(f"[!] Page {page_num + 1}, Image {image_id} is a single color and will be discarded.")
                continue  # Skip to the next image

            # Filter 3: Check if the image matches any reference image
            is_duplicate = False
            for reference_image in reference_images:
                if are_images_equal(extracted_image.convert("RGB"), reference_image):
                    print(f"[!] Page {page_num + 1}, Image {image_id} matches a reference image and will be discarded.")
                    is_duplicate = True
                    page.delete_image(xref)  # Delete the duplicate image from the PDF page
                    break  # Exit the loop as a duplicate is found

            if is_duplicate:
                continue  # Skip to the next image if a duplicate is detected

            # Image has passed all filters
            print(f"[OK] Page {page_num + 1}, Image {image_id} passed all filters and will be retained.")

    # Save the modified PDF with optimizations (garbage collection and compression)
    pdf_document.save(output_pdf_path, garbage=4, deflate=True)
    pdf_document.close()

    # Return the path to the modified PDF and the flag for table presence
    return output_pdf_path, contains_tables


# Example usage
pdf_path = "documents/rango_paginas.pdf"
clean_pdf(pdf_path)
