# -Voice-Based-Document-QA-System
This repository contains the implementation of Lab Exercise 4 for the MAI 457 - Large Language Model course, focusing on a comparative study of transformer-based language models in a voice-based question-answering (QA) system. The system processes PDF documents in English, Kannada, and French, accepts voice input questions, extracts answers using transformer models, and provides voice output using text-to-speech (TTS).

Project Overview

The system implements a voice-based QA pipeline using three transformer-based models:





FLAN-T5 (English, Foundation Model): A fine-tuned T5 model for instruction-following tasks like QA and summarization.



MuRIL (Kannada, Indic Model): A multilingual BERT model supporting Indic languages, optimized for extractive QA.



mT5 (French, International Model): A multilingual T5 model supporting French, used for generative QA and summarization.

The system:





Accepts PDF documents as input.



Processes voice questions via speech-to-text (STT).



Generates answers using the respective models.



Converts answers to speech using gTTS.



Evaluates performance using BLEU, ROUGE, and human metrics (fluency, coherence, relevance).

Features





PDF Processing: Extracts text from English, Kannada, and French PDFs using PyPDF2.



Voice Input/Output: Uses speech_recognition for STT and gTTS for TTS, supporting English, Kannada, and French.



Question Translation: Translates English questions to Kannada and French using deep-translator.



Model Evaluation: Computes BLEU and ROUGE scores and allows human evaluation via a Streamlit UI.



Comparison Table: Displays performance metrics (language fluency, named entity handling, etc.) in a tabular format.

Installation

Prerequisites





Python 3.7 or higher



A quiet environment for speech recognition



Internet connection for gTTS and model downloads



Text-based PDFs for English, Kannada, and French (sample creation scripts included)

Dependencies

Install the required Python packages:

pip install torch transformers streamlit PyPDF2 speechrecognition deep-translator evaluate pandas nltk gTTS

Optional: OCR for Image-Based PDFs

If your PDFs are image-based, install pytesseract and Tesseract OCR:

pip install pytesseract
# Ubuntu
sudo apt-get install tesseract-ocr
# Windows/macOS: Follow https://github.com/tesseract-ocr/tesseract

Setup





Clone the Repository:

git clone https://github.com/your-username/VoiceBasedQASystem.git
cd VoiceBasedQASystem



Prepare PDFs:





Use the provided scripts to create text-based PDFs:





generate_english_doc.py: For English PDF (Karnataka culture).



generate_kannada_doc.py: For Kannada PDF.



generate_french_doc.py: For French PDF.



Alternatively, ensure your PDFs have extractable text (test with pdftotext or a PDF viewer).



Run the Streamlit App:

streamlit run main.py

Usage





Launch the App:





Open the Streamlit app in your browser (typically http://localhost:8501).



Upload PDFs:





Upload English, Kannada, and French PDFs via the UI.



Record Question:





Click the "Record Question" button and speak clearly (e.g., "Summarize the document").



The system translates the English question to Kannada and French.



View Results:





Answers are displayed with BLEU/ROUGE scores.



Audio outputs are played via gTTS.



Rate each model for fluency, coherence, and relevance in the UI.



Compare Models:





A comparison table and visualizations show automatic and human evaluation metrics.

Sample Questions





English: "Summarize the document" or "What is the capital of Karnataka?"



Kannada: "ದಾಖಲೆಯನ್ನು ಸಂಕ್ಷಿಪ್ತಗೊಳಿಸಿ" or "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"



French: "Résumer le document" or "Quelle est la capitale de la France?"

Code Structure





main.py: Main Streamlit app implementing the QA system.



generate_english_doc.py: Script to create an English PDF about Karnataka culture.



generate_kannada_doc.py: Script to create a Kannada PDF.



generate_french_doc.py: Script to create a French PDF.



README.md: This file.

Evaluation Metrics





Automatic Metrics:





BLEU: Measures answer overlap with reference texts.



ROUGE-L: Assesses longest common subsequence for answer quality.



Human Metrics:





Fluency: Clarity and naturalness of the answer (1–5).



Coherence: Logical structure of the answer (1–5).



Relevance: Answer alignment with the question (1–5).



Speech-to-Text Accuracy: Quality of voice recognition.



Text-to-Speech Quality: Clarity of audio output.

Troubleshooting





Empty Answers:





Ensure PDFs have extractable text (test with extract_text_from_pdf function).



Use OCR (pytesseract) for image-based PDFs.



Speech Recognition Errors:





Test in a quiet environment.



Manually set questions in st.session_state["questions"] for debugging.



mT5 Special Tokens:





If <extra_id_0> appears, verify clean_answer function and skip_special_tokens=True.



Low BLEU/ROUGE Scores:





Update references dictionary in main.py to match expected outputs.



TTS Issues:





Ensure internet connectivity for gTTS.



Test with shorter answers if audio is garbled.

Submission





Save the main script as Astro_S01_Lab4.py.



Upload to Google Classroom with PDFs and screenshots of the comparison table.



Include human evaluation ratings from the Streamlit UI.

License

This project is for educational purposes and aligns with the MAI 457 course requirements. No additional license is provided.

Acknowledgments





Hugging Face for transformers and pre-trained models.



Google for gTTS and deep-translator.



Streamlit for the interactive UI.
