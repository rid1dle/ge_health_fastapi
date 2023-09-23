import PyPDF2
import os
import csv


def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)  # PdfFileReader is replaced with PdfReader
        text = ""
        for (
            page
        ) in pdf_reader.pages:  # Use .pages instead of getNumPages() and getPage(int)
            text += page.extract_text()
        return text


def save_text_to_file(text):
    csv_file = "value.csv"
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        # Write the header only if the file didn't exist
        if not file_exists:
            writer.writerow(["text"])
        # Write the extracted text
        writer.writerow([text])
    print("Text extracted and saved to 'value.csv' successfully!")


if __name__ == "__main__":
    pdf_file_path = (
        "./pdfs/pdf1.pdf"  # Example path, please replace with your actual file path
    )
    extracted_text = extract_text_from_pdf(pdf_file_path)
    save_text_to_file(extracted_text)
