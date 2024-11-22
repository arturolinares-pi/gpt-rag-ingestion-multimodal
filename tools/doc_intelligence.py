import os
import time
import json
import logging
import requests
from urllib.parse import urlparse, unquote
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from pydantic import BaseModel, Field
import re
from azure.ai.documentintelligence.models import AnalyzeResult
from PIL import Image
import pymupdf
import mimetypes
import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from azure.identity import get_bearer_token_provider

class DocumentIntelligenceClient:
    """
    A client for interacting with Azure's Document Intelligence service.

    Attributes:
        service_name (str): The name of the Azure Document Intelligence service.
        api_version (str): The API version to use for the service.
        network_isolation (bool): Flag to indicate if network isolation is enabled.

    Methods:
        analyze_document(filepath, model):
            Analyzes a document using the specified model.
    """
    def __init__(self, document_filename=""):
        """
        Initializes the DocumentIntelligence client.

        Parameters:
        document_filename (str, optional): Additional attribute for improved log traceability.
        """
        self.document_filename = f"[{document_filename}]" if document_filename else ""
        
        # ai service resource name
        self.service_name = os.environ['AZURE_FORMREC_SERVICE']
        
        # API configuration
        self.DOCINT_40_API = '2023-10-31-preview'
        self.DEFAULT_API_VERSION = '2023-07-31'
        self.api_version = os.getenv('FORM_REC_API_VERSION', os.getenv('DOCINT_API_VERSION', self.DEFAULT_API_VERSION))
        self.docint_40_api = self.api_version >= self.DOCINT_40_API
                
        # Network isolation
        network_isolation = os.getenv('NETWORK_ISOLATION', self.DEFAULT_API_VERSION)
        self.network_isolation = True if network_isolation.lower() == 'true' else False

        self.aoai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        self.openai_service_name = os.getenv('AZURE_OPENAI_SERVICE_NAME')
        self.openai_api_base = f"https://{self.openai_service_name}.openai.azure.com"
        self.openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        self.openai_embeddings_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
        self.openai_gpt_deployment = os.getenv('AZURE_OPENAI_CHATGPT_DEPLOYMENT')
        self.max_retries = 10  # Maximum number of retries for rate limit errors


        # Supported extensions
        self.file_extensions = [
            "pdf",
            "bmp",
            "jpeg",
            "png",
            "tiff"
        ]
        self.ai_service_type = "formrecognizer"
        self.output_content_format = ""
        self.docint_features = "" 
        self.analyse_output_options = ""
        self.MAX_TOKENS = 2000
        
        if self.docint_40_api:
            self.ai_service_type = "documentintelligence"
            self.file_extensions.extend(["docx", "pptx", "xlsx", "html"])
            self.output_content_format = "markdown"            
            self.analyse_output_options = "figures"

    def download_blob_to_file(self, blob_service_client: BlobServiceClient, container_name, blob_name, filepath="/tmp"):
        """
        Downloads a blob from an Azure Blob Storage container to a specified local directory.

        Parameters:
        - blob_service_client (BlobServiceClient): An authenticated BlobServiceClient instance.
        - container_name (str): The name of the Azure Blob Storage container.
        - blob_name (str): The name of the blob to download.
        - filepath (str): The local directory path to save the downloaded file (default is "/tmp").

        Returns:
        - str: The full path of the downloaded file.

        Notes:
        - Creates the directory specified in `filepath` if it does not already exist.
        - Logs the download status to indicate the start of the download process.
        """
        logging.info(f"[docintelligence] {blob_name} Downloading file...")

        # Define the full file path for the downloaded file
        full_path = os.path.join(filepath, blob_name)

        # Create the directory if it does not exist
        os.makedirs(filepath, exist_ok=True)

        # Get the blob client for the specified blob and initiate the download
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download the blob content to a local file
        with open(full_path, mode="wb") as local_file:
            download_stream = blob_client.download_blob()
            local_file.write(download_stream.readall())
        logging.info(f"[docintelligence] File downloaded to: {full_path}")

    def _get_file_extension(self, filepath):
        # Split the filepath at '?' and take the first part
        clean_filepath = filepath.split('?')[0]
        # Split the clean filepath at '.' and take the last part
        return clean_filepath.split('.')[-1]

    def _get_content_type(self, file_ext):
        extensions = {
            "pdf": "application/pdf", 
            "bmp": "image/bmp",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "tiff": "image/tiff",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "html": "text/html" 
        }
        if file_ext in extensions:
            return extensions[file_ext]
        else:
            return "application/octet-stream"


    def crop_image_from_image(self, image_path, page_number, bounding_box):
        """
        Crops a region from an image file based on a bounding box. Supports multi-page TIFF images.

        Parameters:
        - image_path (str): Path to the image file.
        - page_number (int): The page number of the image to crop (only applicable for TIFF format).
        - bounding_box (tuple): A tuple of (left, upper, right, lower) coordinates defining the region to crop.

        Returns:
        - Image: A PIL Image object of the cropped area.

        Notes:
        - For TIFF images, the specified page is loaded based on `page_number`.
        """
        # Open the image file
        with Image.open(image_path) as img:
            # If the image is in TIFF format and has multiple pages, load the specified page
            if img.format == "TIFF":
                img.seek(page_number)
                img = img.copy()  # Copy the image to avoid issues with file handles

            # Crop the image using the specified bounding box coordinates
            cropped_image = img.crop(bounding_box)
            return cropped_image

    def crop_image_from_pdf_page_entire_page(self, pdf_path, page_number):
        """
        Extracts an entire page from a PDF as an image.

        Parameters:
        - pdf_path (str): Path to the PDF file.
        - page_number (int): The page number to extract (0-indexed).

        Returns:
        - Image: A PIL Image object of the entire page.

        Notes:
        - The page is rendered at a resolution of 300 DPI for high-quality output.
        """
        # Open the PDF document
        doc = pymupdf.open(pdf_path)

        # Load the specified page
        page = doc.load_page(page_number)

        # Define the rectangle that represents the full page dimensions
        rect = page.rect  # Full-page rectangle coordinates (x0, y0, x1, y1)

        # Render the page to an image at 300 DPI resolution for better quality
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72))

        # Convert the pixmap to a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Close the PDF document to release resources
        doc.close()

        return img

    def crop_image_from_pdf_page(self, pdf_path, page_number, bounding_box):

        """
        Crops a specified region from a given page in a PDF file and returns it as an image.

        Parameters:
        - pdf_path (str): Path to the PDF file.
        - page_number (int): The page number to crop from (0-indexed).
        - bounding_box (tuple): A tuple of (x0, y0, x1, y1) coordinates representing the bounding box.

        Returns:
        - Image: A PIL Image object of the cropped area.
        """
        # Open the PDF document.
        doc = pymupdf.open(pdf_path)

        # Load the specified page.
        page = doc.load_page(page_number)

        # Define the bounding box with scaling for the PDF coordinates (DPI is scaled to 300).
        bbx = [coord * 72 for coord in bounding_box]
        rect = pymupdf.Rect(bbx)

        # Create a pixmap (image) from the selected region with 300 DPI.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), clip=rect)

        # Convert the pixmap to a PIL Image.
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Close the document after processing.
        doc.close()

        return img

    def crop_image_from_file(self, file_path, page_number, bounding_box):
        """
        Crops an image from a file, which could be a PDF, TIFF, or other supported image file.

        Parameters:
        - file_path (str): The path to the file.
        - page_number (int): The page number (for multi-page files like PDF and TIFF, 0-indexed).
        - bounding_box (tuple): A tuple of (x0, y0, x1, y1) coordinates for the bounding box.

        Returns:
        - Image: A PIL Image object of the cropped area.
        """
        # Guess the MIME type based on the file extension.
        mime_type = mimetypes.guess_type(file_path)[0]

        # Determine the method based on the file type (PDF or other image).
        if mime_type == "application/pdf":
            # If PDF, use the PDF cropping function.
            return self.crop_image_from_pdf_page(file_path, page_number, bounding_box)
        else:
            # For non-PDF files, use the general image cropping function.
            return self.crop_image_from_image(file_path, page_number, bounding_box)

    # Function to encode a local image into data URL
    def local_image_to_data_url(self, image_path):
        """
        Encodes a local image file as a data URL.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - str: The image as a base64-encoded data URL.
        """
        # Guess the MIME type based on the file extension.
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Use a default MIME type if none is found

        # Read and encode the image file in base64.
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct and return the data URL.
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def verify_text_with_gpt(self, deployment_name, openai_api_version, image_path, content):
        """
        Verifies and corrects text in an image using the GPT model, ensuring the text is accurately extracted
        from the image and formatted correctly.

        Parameters:
        - deployment_name (str): The name of the OpenAI deployment.
        - openai_api_version (str): The version of the OpenAI API.
        - image_path (str): The path to the image file.
        - content (str): The textual content extracted from the image for verification.

        Returns:
        - text_description (ActionEvent): An object describing the suggested action:
            - "replace" with a corrected text if there are errors.
            - "continue" if the text is correct and requires no action.
        """

        # Define the data model for content replacement response.
        class ReplaceContent(BaseModel):
            original_text: str = Field(
                description="The original text of the section that will be replaced, keeping its original structure."
            )
            replace_text: str = Field(
                description="The corrected text, only changing the letters without altering the original structure."
            )

        # Define the data model for the action event.
        class ActionEvent(BaseModel):
            content: list[ReplaceContent] = Field(
                description="A list of ReplaceContent objects if the action is 'replace'. Empty if the action is 'continue'."
            )
            action: str = Field(
                description="Action to perform, which can be 'replace' if corrections are needed or 'continue' if the text is correct."
            )

        # Set up the Azure authentication token provider.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        # Initialize the Azure OpenAI client with the required API version and endpoint.
        client = AzureOpenAI(
            api_version=openai_api_version,
            azure_endpoint=self.openai_api_base,
            azure_ad_token_provider=token_provider,
            max_retries=self.max_retries
        )

        # Convert the local image to a data URL format for transmission.
        data_url = self.local_image_to_data_url(image_path)

        # Prepare the request message, providing instructions and including both the image and its content.
        response = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Verify the information."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": '''Analyze the provided image, which is a screenshot of pages from a PDF, to verify if the textual content is correctly written.

        Ensure that any errors or inconsistencies in the text are corrected while maintaining the original structure and meaning. Do not consider accents or diacritics such as tildes to be errors. Also, there is no need to analyze tables or images within the provided image.

        # Output Format

        - If there are errors in the text:
          Return a JSON response in the following format, specifying the error and its correction:
          ```json
          {
              "action": "replace",
              "content": [
                  {
                      "original_text": "[original section of the text]",
                      "replace_text": "[corrected section of the text]"
                  }
              ]
          }
          ```

        - If the text is correct:
          Return the following JSON response:
          ```json
          {
              "action": "continue",
              "content": []
          }
          ```

        # Examples

        - Example 1:
          - Input: (Snapshot of text) "The quick broown fox jumps oover the lazy dog."
          - Output:
          ```json
          {
              "action": "replace",
              "content": [
                  {
                      "original_text": "The quick broown fox jumps oover the lazy dog.",
                      "replace_text": "The quick brown fox jumps over the lazy dog."
                  }
              ]
          }
          ```

        - Example 2:
          - Input: (Snapshot of text) "All is well that ends well."
          - Output:
          ```json
          {
              "action": "continue",
              "content": []
          }
          ```

        # Notes

        - Focus on verifying and correcting the text only.
        - Ignore any tables or non-textual elements within the image.
        - Accents and diacritics should not be considered as errors.'''
                    },
                    {
                        "type": "text",
                        "text": f"Content:\n{content}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]}
            ],
            response_format=ActionEvent,
            max_tokens=self.MAX_TOKENS
        )

        # Parse the response to determine the suggested action.
        message = response.choices[0].message
        if message.parsed:
            # Output the action if parsing was successful.
            print(message.parsed.action)
        else:
            # If parsing failed, print the refusal message.
            print(message.refusal)

        # Return the parsed response as the text description.
        text_description = message.parsed

        return text_description

    def understand_table_with_gpt(self, deployment_name, openai_api_version, image_path, content):
        """
        Analyzes a table in an image using the GPT model to determine if it is a standard, word-processed table
        or an image-based table and suggests an action accordingly.

        Parameters:
        - deployment_name (str): The name of the OpenAI deployment.
        - openai_api_version (str): The version of the OpenAI API.
        - image_path (str): The path to the image file.
        - content (str): The extracted content of the table.

        Returns:
        - table_description (ActionEvent): An object describing the suggested action:
            - "delete" for image-based tables.
            - "replace" with corrected content for word-processed tables with errors.
            - "continue" if the table is correct and requires no action.
        """

        # Define the data model for the action response.
        class ActionEvent(BaseModel):
            action: str = Field(
                description="Action to perform, which can be 'delete', 'replace', or 'continue' when everything is correct."
            )
            content: str = Field(
                description="Modified content when the action is 'replace'. Empty if the action is 'delete' or 'continue'."
            )

        # Set up the Azure authentication token provider.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        # Initialize the Azure OpenAI client with the required API version and endpoint.
        client = AzureOpenAI(
            api_version=openai_api_version,
            azure_endpoint=self.openai_api_base,
            azure_ad_token_provider=token_provider,
            max_retries=self.max_retries
        )

        # Convert the local image to a data URL format for transmission.
        data_url = self.local_image_to_data_url(image_path)

        # Prepare the request message, providing instructions and including both the image and its content.
        response = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Verify the information."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": '''Analyze the provided content and image to determine if the table in the image appears to be a standard table created in a word processor or part of the image itself.

        # Steps

        1. **Identifying Tables in Images vs. Text Document:**
           - Examine the visual appearance of the table.
           - Identify clean structures, aligned text, visible gridlines, and consistent cell borders for standard tables.
           - Look for uneven borders, graphical elements, skewing, or misaligned text for image-based tables.
           - Evaluate the table's placement. Neatly placed with surrounding text indicates a word-processed table.

        2. **Determining Action Based on Table Type:**
           - **For image-based tables:**
             - **Return:**
               - `{"action": "delete"}`

           - **For word-processed tables:**
             - Examine for errors or inconsistencies.
             - **If errors are present:**
               - Correct the errors while retaining the original structure.
               - **Return:**
                 - `{"action": "replace", "content": "[corrected table in text]"}` (Insert the corrected table in place.)
             - **If the table is correct:**
               - **Return:**
                 - `{"action": "continue"}`

        # Output Format

        - JSON object describing the action:
          - For image-based tables: `{"action": "delete"}`
          - For corrected tables: `{"action": "replace", "content": "[corrected table in text]"}` (Replace with the actual corrected table content in place of the placeholder.)
          - For correct tables: `{"action": "continue"}`

        # Notes

        - Thoroughly analyze table characteristics to distinguish between image-based and word-processed formats.
        - Maintain document professionalism and accuracy in output.
        '''
                    },
                    {
                        "type": "text",
                        "text": f"Content:\n{content}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]}
            ],
            response_format=ActionEvent,
            max_tokens=self.MAX_TOKENS
        )

        # Parse the response to determine the suggested action.
        message = response.choices[0].message
        if message.parsed:
            # Output the action if parsing was successful.
            print(message.parsed.action)
        else:
            # If parsing failed, print the refusal message.
            print(message.refusal)

        # Return the parsed response as the table description.
        table_description = message.parsed

        return table_description

    def understand_image_with_gpt(self, deployment_name, openai_api_version, image_path, caption):
        """
        Generates a description for an image using the GPT model.

        Parameters:
        - deployment_name (str): The name of the OpenAI deployment.
        - openai_api_version (str): The version of the OpenAI API.
        - image_path (str): The path to the image file.
        - caption (str): The caption or title of the image (can be an empty string if none exists).

        Returns:
        - img_description (str): The generated description for the image.

        Description:
        This function uses Azure OpenAI to generate a detailed description of an image.
        Both the image and an optional caption are sent to the GPT model to get a description
        in Spanish of the image.
        """
        # Get the token provider for Azure authentication.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        # Create the Azure OpenAI client with the specified configuration.
        client = AzureOpenAI(
            api_version=openai_api_version,
            azure_endpoint=self.openai_api_base,
            azure_ad_token_provider=token_provider,
            max_retries=self.max_retries
        )

        # Convert the local image to a data URL format for transmission.
        data_url = self.local_image_to_data_url(image_path)

        # Prepare messages for the GPT request, including the image and caption, if available.
        if caption != "":
            # Create a chat request with a message that includes the image caption.
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond in Spanish."},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image (note: it has a caption: {caption}):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]}
                ],
                max_tokens=self.MAX_TOKENS
            )
        else:
            # Create a chat request without including the image caption.
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always respond in Spanish."},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "Describe this image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]}
                ],
                max_tokens=self.MAX_TOKENS
            )

        # Extract the image description from the GPT response.
        img_description = response.choices[0].message.content

        return img_description

    def update_figure_description(self, md_content, img_description, idx):
        """
        Updates the figure description in the Markdown content.

        Args:
            md_content (str): The original Markdown content.
            img_description (str): The new description for the image.
            idx (int): The index of the figure.

        Returns:
            str: The updated Markdown content with the new figure description.

        Example:
            Given a Markdown content with figures in the format:
            ![](figures/{index})
            </figure>

            This method will insert a new comment `<!-- FigureContent="{description}" -->`
            right after the figure placeholder, within the specified figure block.
        """

        # Define the starting substring pattern for the figure to update.
        start_substring = f"![](figures/{idx})"
        # Define the ending substring pattern that marks the end of the figure block.
        end_substring = "</figure>"
        # The new comment string to insert with the image description.
        new_string = f"<!-- FigureContent=\"{img_description}\" -->"

        # Initialize new_md_content to md_content in case no updates are made.
        new_md_content = md_content

        # Find the start index of the substring for the figure to replace.
        start_index = md_content.find(start_substring)
        if start_index != -1:  # Proceed only if start_substring is found
            # Move the index to the end of start_substring to begin insertion.
            start_index += len(start_substring)
            # Find the end index of the figure block.
            end_index = md_content.find(end_substring, start_index)

            # Proceed only if end_substring is found after start_substring.
            if end_index != -1:
                # Construct the updated Markdown content with the new description.
                # Insert new_string between start_index and end_index.
                new_md_content = (
                        md_content[:start_index] + new_string + md_content[end_index:]
                )

        # Return the updated Markdown content with the modified figure description.
        return new_md_content

    # helper functions

    def get_words(self, page, line):
        result = []
        for word in page.words:
            if self._in_span(word, line.spans):
                result.append(word)
        return result

    def _in_span(self, word, spans):
        for span in spans:
            if word.span.offset >= span.offset and (
                    word.span.offset + word.span.length
            ) <= (span.offset + span.length):
                return True
        return False

    def analyze_layout(self, document, input_file_path):
        global file_name_without_extension, cropped_image_filename, region
        flag = True
        # Obtener solo la ruta del directorio
        output_folder = os.path.dirname(input_file_path)
        # Define the base directory

        # Define the subdirectories to be created
        subdirectories = ["images", "tables", "text"]

        # Create the base directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create each subdirectory inside the base directory
        for subdir in subdirectories:
            os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)

        result = AnalyzeResult(document)
        md_content = result.content
        if not flag:
            # Expresión regular para encontrar todas las tablas <table>...</table>
            md_content = re.sub(r'<table>.*?</table>', '', md_content, flags=re.DOTALL)
            print("Se han eliminado todas las tablas del contenido Markdown.")

        if result.tables:
            for table_idx, table in enumerate(result.tables):
                table_content = ""
                print(f"Figure #{table_idx} has the following spans: {table.spans}")
                for i, span in enumerate(table.spans):
                    print(f"Span #{i}: {span}")

                    table_content += result.content[span.offset:span.offset + span.length]
                    # Definir el substring donde quieres reemplazar el contenido
                    start_substring = (f"![](table/{table_idx})"
                                       f"<table>")
                    # Reemplazar la etiqueta <figure> por el valor de start_substring
                    # Solo se reemplaza el inicio de la etiqueta <figure>
                    table_content_replace = table_content.replace("<table>", start_substring, 1)
                    md_content = md_content.replace(table_content, table_content_replace)
                    table_content = md_content
                    nu_page = table.bounding_regions[0].page_number - 1
                    cropped_page = self.crop_image_from_pdf_page_entire_page(input_file_path, nu_page)

                    # Get the base name of the file
                    base_name = os.path.basename(input_file_path)
                    # Remove the file extension
                    file_name_without_extension = os.path.splitext(base_name)[0]

                    output_file = f"tables/{file_name_without_extension}_cropped_page_table_{nu_page}.png"
                    cropped_page_filename = os.path.join(output_folder, output_file)

                    cropped_page.save(cropped_page_filename)
                    print(f"\tFigure {table_idx} cropped and saved as {cropped_page_filename}")

                    table_validation = self.understand_table_with_gpt(self.openai_gpt_deployment,
                                                                      self.openai_api_version, cropped_page_filename,
                                                                      table_content_replace
                                                                      )

                    print(table_validation.action)
                    # Evaluar la acción obtenida de la validación de la tabla
                    if table_validation.action == "continue":
                        # Si la acción es "continue", pasar a la siguiente iteración
                        continue

                    elif table_validation.action == "delete":
                        # Si la acción es "delete", reemplazar el contenido de la tabla por vacío
                        md_content = md_content.replace(table_content_replace, "")
                        print(f"Table {table_idx} has been deleted.")

                    elif table_validation.action == "replace":
                        # Si la acción es "replace", reemplazar el contenido de la tabla con el texto corregido
                        # Reemplazar tod0 el contenido desde <table> hasta </table> con el contenido corregido
                        start_idx = md_content.find("<table>", md_content.find(table_content_replace))
                        end_idx = md_content.find("</table>", start_idx) + len("</table>")
                        md_content = md_content[:start_idx] + table_validation.content + md_content[end_idx:]
                        print(f"Table {table_idx} has been replaced with corrected content.")

                print(f"Original figure content in markdown: {table_content}")

        metadata = []

        if result.figures:
            print("Figures:")
            for idx, figure in enumerate(result.figures):
                figure_content = ""
                img_description = ""
                print(f"Figure #{idx} has the following spans: {figure.spans}")
                for i, span in enumerate(figure.spans):
                    print(f"Span #{i}: {span}")

                    figure_content += result.content[span.offset:span.offset + span.length]
                    # Definir el substring donde quieres reemplazar el contenido
                    start_substring = f"![](figures/{idx})"
                    # Reemplazar la etiqueta <figure> por el valor de start_substring
                    # Solo se reemplaza el inicio de la etiqueta <figure>
                    figure_content_replace = figure_content.replace("<figure>", start_substring, 1)
                    md_content = md_content.replace(figure_content, figure_content_replace)
                    figure_content = md_content

                print(f"Original figure content in markdown: {figure_content}")

                # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
                if figure.caption:
                    caption = figure.caption.content
                    caption_region = figure.caption.bounding_regions
                    print(f"\tCaption: {figure.caption.content}")
                    print(f"\tCaption bounding region: {caption_region}")
                    for region in figure.bounding_regions:
                        if region not in caption_region:
                            print(f"\tFigure body bounding regions: {region}")
                            # To learn more about bounding regions, see https://aka.ms/bounding-region
                            boundingbox = (
                                region.polygon[0],  # x0 (left)
                                region.polygon[1],  # y0 (top)
                                region.polygon[4],  # x1 (right)
                                region.polygon[5]  # y1 (bottom)
                            )
                            print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")
                            cropped_image = self.crop_image_from_file(input_file_path, region.page_number - 1,
                                                                      boundingbox)  # page_number is 1-indexed

                            # Get the base name of the file
                            base_name = os.path.basename(input_file_path)
                            # Remove the file extension
                            file_name_without_extension = os.path.splitext(base_name)[0]

                            output_file = f"images/{file_name_without_extension}_cropped_image_{idx}.png"
                            cropped_image_filename = os.path.join(output_folder, output_file)

                            cropped_image.save(cropped_image_filename)
                            print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                            img_description += self.understand_image_with_gpt(self.openai_gpt_deployment,
                                                                              self.openai_api_version,
                                                                              cropped_image_filename,
                                                                              figure.caption.content)
                            print(f"\tDescription of figure {idx}: {img_description}")
                else:
                    caption = ""
                    print("\tNo caption found for this figure.")
                    for region in figure.bounding_regions:
                        print(f"\tFigure body bounding regions: {region}")
                        # To learn more about bounding regions, see https://aka.ms/bounding-region
                        boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top
                            region.polygon[4],  # x1 (right)
                            region.polygon[5]  # y1 (bottom)
                        )
                        print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")

                        cropped_image = self.crop_image_from_file(input_file_path, region.page_number - 1,
                                                                  boundingbox)  # page_number is 1-indexed

                        # Get the base name of the file
                        base_name = os.path.basename(input_file_path)
                        # Remove the file extension
                        file_name_without_extension = os.path.splitext(base_name)[0]

                        output_file = f"images/{file_name_without_extension}_cropped_image_{idx}.png"
                        cropped_image_filename = os.path.join(output_folder, output_file)
                        # cropped_image_filename = f"data/cropped/image_{idx}.png"
                        cropped_image.save(cropped_image_filename)
                        print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                        img_description += self.understand_image_with_gpt(self.openai_gpt_deployment,
                                                                          self.aoai_api_version, cropped_image_filename,
                                                                          "")
                        print(f"\tDescription of figure {idx}: {img_description}")

                # Obtener dimensiones de la imagen
                img = Image.open(cropped_image_filename)
                width, height = img.size
                print(f"\tImage dimensions: {width}x{height}")
                print(f"\tDescription of figure {idx}: {img_description}")

                # Crear los metadatos para la figura
                figure_metadata = {
                    "image_path": cropped_image_filename,
                    "description": img_description,
                    "page_number": region.page_number,
                    "width": width,
                    "height": height,
                    "caption": caption
                }
                metadata.append(figure_metadata)

                # replace_figure_description(figure_content, img_description, idx)
                md_content = self.update_figure_description(md_content, img_description, idx)
        output_metadata_file = os.path.join(output_folder, f"images/{file_name_without_extension}_metadata.json")
        with open(output_metadata_file, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        print(f"Metadata saved in {output_metadata_file}")

        paginas = md_content.split("<!-- PageBreak -->")

        total_paginas = len(result.pages)

        for pagina in range(total_paginas):
            # Llamar a la función que recorta la imagen de la página
            pagina_original = paginas[pagina]
            cropped_page = self.crop_image_from_pdf_page_entire_page(input_file_path, pagina)
            # Get the base name of the file
            base_name = os.path.basename(input_file_path)
            # Remove the file extension
            file_name_without_extension = os.path.splitext(base_name)[0]

            # Crear el nombre del archivo de imagen para la página recortada
            output_file = f"text/{file_name_without_extension}_cropped_page_text_{pagina + 1}.png"
            cropped_page_filename = os.path.join(output_folder, output_file)

            # Guardar la imagen recortada
            cropped_page.save(cropped_page_filename)

            # Llamar a la función de validación del texto en la página actual
            text_validation = self.verify_text_with_gpt(self.openai_gpt_deployment,
                                                        self.openai_api_version, cropped_page_filename,
                                                        pagina_original)
            if text_validation.action == "continue":
                print("continue")

            elif text_validation.action == "replace":
                print("replace")

                # Iterar sobre cada elemento de la lista 'content'
                for replacement in text_validation.content:
                    original_text = replacement.original_text
                    replace_text = replacement.replace_text

                    # Hacer el reemplazo en 'md_content' del original_text por el replace_text
                    md_content = md_content.replace(original_text, replace_text)

                print("El contenido ha sido reemplazado.")

        nombre_archivo = f"{file_name_without_extension}.md"
        ruta_md = os.path.join(output_folder, nombre_archivo)
        with open(ruta_md, "w", encoding="utf-8") as file:
            file.write(md_content)

        print(f"Contenido guardado en {nombre_archivo}")

        return md_content

    def analyze_document(self, file_url, model='prebuilt-layout'):
        """
        Analyzes a document using the specified model.

        Args:
            file_url (str): The url to the document to be analyzed.
            model (str): The model to use for document analysis.

        Returns:
            tuple: A tuple containing the analysis result and any errors encountered.
        """
        result = {}
        errors = []

        file_ext = self._get_file_extension(file_url)

        if file_ext in ["pdf"]:
            self.docint_features = "ocr.highResolution"

        # Set request endpoint
        request_endpoint = f"https://{self.service_name}.cognitiveservices.azure.com/{self.ai_service_type}/documentModels/{model}:analyze?api-version={self.api_version}"
        if self.docint_features:
            request_endpoint += f"&features={self.docint_features}" 
        if self.output_content_format:
            request_endpoint += f"&outputContentFormat={self.output_content_format}"
        if self.analyse_output_options:
            request_endpoint += f"&output={self.analyse_output_options}"

        # Set request headers
        token = DefaultAzureCredential().get_token("https://cognitiveservices.azure.com/.default")

        headers = {
                    "Content-Type": self._get_content_type(file_ext),
                    "Authorization": f"Bearer {token.token}",
                    "x-ms-useragent": "gpt-rag/1.0.0"
                }            
        parsed_url = urlparse(file_url)
        account_url = parsed_url.scheme + "://" + parsed_url.netloc
        container_name = parsed_url.path.split("/")[1]
        url_decoded = unquote(parsed_url.path)
        blob_name = url_decoded[len(container_name) + 2:]
        file_ext = blob_name.split(".")[-1]

        logging.info(f"[docintelligence]{self.document_filename} Connecting to blob.")

        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        temp_path = "/tmp"  # Usa el directorio temporal para escritura

        self.download_blob_to_file(blob_service_client, container_name, blob_name, temp_path)

        logging.info(f"[docintelligence]{self.document_filename} Archivo descargado.")

        blob_error = None

        try:
            data = blob_client.download_blob().readall()
            response = requests.post(request_endpoint, headers=headers, data=data)
        except requests.exceptions.ConnectionError:
            logging.info(f"[docintelligence]{self.document_filename} Connection error, retrying in 10 seconds...")
            time.sleep(10)
            try:
                data = blob_client.download_blob().readall()
                response = requests.post(request_endpoint, headers=headers, data=data)
            except Exception as e:
                blob_error = e
        except Exception as e:
            blob_error = e

        if blob_error:
            error_message = f"Blob client error when reading from blob storage. {blob_error}"
            logging.info(f"[docintelligence]{self.document_filename} {error_message}")
            errors.append(error_message)
            return result, errors

        error_messages = {
            404: "Resource not found, please verify your request url. The Doc Intelligence API version you are using may not be supported in your region.",
        }
        
        if response.status_code in error_messages or response.status_code != 202:
            error_message = error_messages.get(response.status_code, f"Doc Intelligence request error, code {response.status_code}: {response.text}")
            logging.info(f"[docintelligence]{self.document_filename} {error_message}")
            logging.info(f"[docintelligence]{self.document_filename} filepath: {file_url}")
            errors.append(error_message)
            return result, errors

        get_url = response.headers["Operation-Location"]
        result_headers = headers.copy()
        result_headers["Content-Type"] = "application/json-patch+json"

        while True:
            result_response = requests.get(get_url, headers=result_headers)
            result_json = json.loads(result_response.text)

            if result_response.status_code != 200 or result_json["status"] == "failed":
                error_message = f"Doc Intelligence polling error, code {result_response.status_code}: {response.text}"
                logging.info(f"[docintelligence]{self.document_filename} {error_message}")
                errors.append(error_message)
                break

            if result_json["status"] == "succeeded":
                result = result_json['analyzeResult']
                break

            time.sleep(2)

        # # Define la ruta completa del archivo
        full_path = os.path.join(temp_path, blob_name)

        result = self.analyze_layout(result, full_path)
        # result=result["content"]

        return result, errors