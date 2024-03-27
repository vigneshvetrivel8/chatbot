from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import csv
from io import BytesIO
from pdfminer.high_level import extract_text_to_fp
import docx
from transformers import pipeline

# Load environment variables from .env file (if any)
load_dotenv()

class Response(BaseModel):
    result: str | None

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=Response)
async def predict(question: str = Form(...), file: UploadFile = File(...)) -> Any:
    try:
        # Read the contents of the uploaded file
        contents = await file.read()
        file_extension = file.filename.split(".")[-1]
        
        # Handle different file types
        if file_extension == "pdf":
            extracted_text = extract_text_from_pdf(contents)
        elif file_extension == "txt":
            extracted_text = extract_text_from_txt(contents)
        elif file_extension == "csv":
            extracted_text = extract_text_from_csv(contents)
        elif file_extension == "docx":
            extracted_text = extract_text_from_docx(contents)
        else:
            return {"result": "Unsupported file type."}
        
        # Perform question answering or summarization based on the question provided
        if "summary" in question.lower() or "summari" in question.lower():
            summary = generate_summary(f"{extracted_text}")
            return {"result": summary}
        else:
            answer = get_answer(question, f"{extracted_text}")
            return {"result": answer}
    
    except Exception as e:
        # Print the error
        print("Error:", e)
        
        # Return an error message
        return {"result": "Failed to process the request."}

def extract_text_from_pdf(contents):
    # Use PDFMiner to extract text from the PDF
    output = BytesIO()
    extract_text_to_fp(BytesIO(contents), output)
    extracted_text = output.getvalue().decode("utf-8")
    return extracted_text

def extract_text_from_txt(contents):
    # Decode the bytes to string assuming utf-8 encoding
    extracted_text = contents.decode("utf-8")
    return extracted_text

def extract_text_from_csv(contents):
    # Use CSV module to read and extract text from the CSV
    extracted_text = ""
    decoded_contents = contents.decode("utf-8")
    csv_reader = csv.reader(decoded_contents.splitlines(), delimiter=',')
    for row in csv_reader:
        extracted_text += ', '.join(row) + '\n'
    return extracted_text

def extract_text_from_docx(contents):
    # Use python-docx to extract text from the DOCX
    doc = docx.Document(BytesIO(contents))
    extracted_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return extracted_text

def get_answer(question, paragraph):
    # Use a Question Answering model to get the answer
    # qa_pipeline = pipeline("question-answering")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    print("after qa_pipeline")
    answer = qa_pipeline(question=question, context=paragraph)
    print("end")
    return answer["answer"]

def process_text_in_chunks(text, chunk_size, process_function):
    """
    Process the input text in chunks using the specified process function.
    
    Args:
        text (str): The input text to be processed.
        chunk_size (int): The maximum size of each chunk.
        process_function (callable): The function to process each chunk.
        
    Returns:
        str: The aggregated result after processing all chunks.
    """
    # Split the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Process each chunk and collect results
    results = [process_function(chunk) for chunk in chunks]
    
    # Aggregate results
    aggregated_result = "\n".join(results)
    
    return aggregated_result

def summarize_chunk(chunk):
    """
    Summarize a text chunk using the BART model.
    
    Args:
        chunk (str): The input text chunk to be summarized.
        
    Returns:
        str: The summary of the input text chunk.
    """
    # Use the summarization pipeline to generate a summary
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
    summary = summarization_pipeline(chunk, max_length=150, min_length=15, do_sample=False)[0]['summary_text']
    return summary

def generate_summary(paragraph):
    # Use a Summarization model to generate a summary
    processed_text = process_text_in_chunks(paragraph, chunk_size=1000, process_function=summarize_chunk)
    return processed_text