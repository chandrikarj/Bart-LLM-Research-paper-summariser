import streamlit as st
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from PyPDF2 import PdfReader
import re

# Load the pre-trained BART model for summarization
@st.cache_resource
def load_model():
    model_args = Seq2SeqArgs()
    model_args.max_length = 100  # Maximum length of the summary
    model_args.length_penalty = 2.0
    model_args.num_beams = 4  # Beam search for better results

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large-cnn",
        args=model_args,
        use_cuda=False,  # Set to True if you have a GPU
    )
    return model

model = load_model()

# Clean and preprocess the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove non-alphanumeric characters (optional)
    text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Extract text from a PDF and clean it
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    if not text.strip():
        return "No readable text found in the PDF."
    return clean_text(text)

# Truncate text to fit model token limits
def truncate_text(text, max_length=1024):
    return text[:max_length]

# Summarization function
def summarize_text(text):
    summaries = model.predict([text])
    return summaries[0]

# Streamlit app
st.title("Biology Research Summarizer with SimpleTransformers")
uploaded_file = st.file_uploader("Upload a Biology Research Paper (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Reading file..."):
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8").strip()
            text = clean_text(text)
        else:
            st.error("Unsupported file type!")
            st.stop()

        if not text.strip():
            st.error("The uploaded file does not contain readable content.")
            st.stop()

        # Show preview of cleaned text
        st.write("Uploaded File Preview (First 1000 characters):")
        st.text(text[:1000])  # Display as plain text to avoid formatting issues

    if st.button("Summarise"):
        with st.spinner("Summarising..."):
            text = truncate_text(text)  # Truncate to fit token limits
            summary = summarize_text(text)
        st.write("Summary:")
        st.text(summary)  # Display as plain text
