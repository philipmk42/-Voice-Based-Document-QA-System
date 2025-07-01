# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, pipeline
import streamlit as st
import PyPDF2
import speech_recognition as sr
from gtts import gTTS
from deep_translator import GoogleTranslator
import evaluate
import pandas as pd
import nltk
import os
import tempfile
import logging
import re

nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model metadata
MODELS = {
    "FLAN-T5": {
        "model_name": "google/flan-t5-base",
        "type": "seq2seq",
        "language": "English",
        "description": "FLAN-T5 is a fine-tuned T5 model optimized for instruction-following tasks. Trained on a diverse English corpus, it excels in QA, translation, and summarization."
    },
    "MuRIL": {
        "model_name": "google/muril-base-cased",
        "type": "qa",
        "language": "Kannada",
        "description": "MuRIL is a multilingual BERT model supporting Indic languages like Kannada. Trained on Indic corpora, it supports NLP tasks like QA."
    },
    "mT5": {
        "model_name": "google/mt5-base",
        "type": "seq2seq",
        "language": "French",
        "description": "mT5 is a multilingual T5 model supporting French, trained on mC4. It excels in generative tasks like summarization and QA for non-English languages."
    }
}

@st.cache_resource
def load_model(model_info):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_info["type"] == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_info["model_name"])
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_info["model_name"])
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model {model_info['model_name']}: {e}")
        raise

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_file.name}")
            st.error(f"No text extracted from {pdf_file.name}. Ensure the PDF contains selectable text.")
            return ""
        logger.info(f"Extracted {len(text)} characters from {pdf_file.name}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def speech_to_text():
    if "listening" in st.session_state and st.session_state["listening"]:
        return st.session_state.get("question_en", None)
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for question...")
        st.session_state["listening"] = True
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language="en-US")
            st.write(f"Recognized: {text}")
            st.session_state["question_en"] = text
            st.session_state["listening"] = False
            return text
        except sr.WaitTimeoutError:
            st.error("No speech detected within timeout.")
            st.session_state["listening"] = False
            return None
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
            st.session_state["listening"] = False
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            st.session_state["listening"] = False
            return None

def text_to_speech(text, lang="en"):
    if not text.strip():
        text = "Aucune réponse générée." if lang == "fr" else "No answer generated."
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name)
        os.unlink(tmp_file.name)
    except Exception as e:
        st.error(f"TTS error: {e}")
        logger.error(f"TTS error for lang {lang}: {e}")

def translate_text(text, src_lang, dest_lang):
    try:
        translator = GoogleTranslator(source=src_lang, target=dest_lang)
        translated = translator.translate(text)
        if not translated:
            raise ValueError("Translation returned empty result")
        return translated
    except Exception as e:
        st.error(f"Translation error: {e}")
        logger.error(f"Translation error from {src_lang} to {dest_lang}: {e}")
        return text

def is_question_relevant(question, context, lang):
    """Check if the question is relevant to the context."""
    question_lower = question.lower()
    context_lower = context.lower()
    keywords = {
        "English": ["culture", "karnataka", "bengaluru", "literature", "dance", "yaks"],
        "Kannada": ["ಸಂಸ್ಕೃತಿ", "ಕರ್ನಾಟಕ", "ಬೆಂಗಳೂರು", "ಸಾಹಿತ್ಯ", "ನೃತ್ಯ", "ಯಕ್ಷಗಾನ"],
        "French": ["culture", "française", "paris", "littérature", "art", "cuisine", "impressionnisme"]
    }
    return any(keyword in question_lower for keyword in keywords.get(lang, [])) or any(keyword in context_lower for keyword in keywords.get(lang, []))

def clean_answer(answer):
    """Remove special tokens and clean the answer."""
    return re.sub(r'<extra_id_\d+>', '', answer).strip()

def answer_question(model_name, context, question, lang):
    if not context or not question:
        logger.warning("Empty context or question")
        return "Erreur : Contexte ou question vide." if lang == "French" else "Error: Empty context or question."

    if not is_question_relevant(question, context, lang):
        logger.warning(f"Irrelevant question for {lang} context: {question}")
        return "La question ne semble pas pertinente pour le contenu du document." if lang == "French" else "The question does not seem relevant to the document content."

    model_info = MODELS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_info["type"] == "seq2seq":
        model, tokenizer = load_model(model_info)
        model = model.to(device)
        if "summari" in question.lower() or "résumer" in question.lower() or "ಸಂಕ್ಷಿಪ್ತ" in question.lower():
            prompt = (
                f"Résumez le document suivant en quelques phrases en français :\n{context[:1000]}" if lang == "French" else
                f"ಸಂಕ್ಷಿಪ್ತವಾಗಿ ದಾಖಲೆಯನ್ನು ಕೆಲವು ವಾಕ್ಯಗಳಲ್ಲಿ ಸಾರಾಂಶಗೊಳಿಸಿ :\n{context[:1000]}" if lang == "Kannada" else
                f"Summarize the following document in a few sentences:\n{context[:1000]}"
            )
        else:
            prompt = (
                f"Contexte : {context[:1000]}\nQuestion : {question}\nRépondez en français :" if lang == "French" else
                f"ಸಂದರ್ಭ : {context[:1000]}\nಪ್ರಶ್ನೆ : {question}\nಉತ್ತರ ಕನ್ನಡದಲ್ಲಿ :" if lang == "Kannada" else
                f"Context: {context[:1000]}\nQuestion: {question}\nAnswer:"
            )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        try:
            outputs = model.generate(**inputs, max_length=150, num_beams=4, no_repeat_ngram_size=2)
            answer = clean_answer(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        except Exception as e:
            logger.error(f"Error generating answer with {model_name}: {e}")
            answer = "Erreur : Impossible de générer une réponse." if lang == "French" else "Error: Unable to generate answer."
    else:
        qa_pipeline = pipeline("question-answering", model=model_info["model_name"], tokenizer=model_info["model_name"], device=0 if torch.cuda.is_available() else -1)
        try:
            result = qa_pipeline(question=question, context=context[:1000], max_answer_len=100)
            answer = result["answer"]
        except Exception as e:
            logger.error(f"QA pipeline error for {model_name}: {e}")
            answer = "Erreur : Impossible de générer une réponse." if lang == "French" else "Error: Unable to generate answer."
    
    return answer.strip()

# Evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def evaluate_answer(reference, generated):
    if not generated.strip() or not reference.strip() or generated.startswith("Erreur :") or generated.startswith("Error:"):
        return {"bleu": 0.0, "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}}
    
    try:
        bleu_score = bleu.compute(predictions=[generated], references=[[reference]])["bleu"]
    except ZeroDivisionError:
        bleu_score = 0.0
    
    rouge_score = rouge.compute(predictions=[generated], references=[reference])
    return {"bleu": bleu_score, "rouge": rouge_score}

# Main Streamlit App
def main():
    st.title("📚 Voice-Based Document QA System")
    
    st.markdown("""
    ### 🚀 Project Overview
    This app implements a voice-based QA system using:
    - **FLAN-T5** (English, Foundation)
    - **MuRIL** (Kannada, Indic)
    - **mT5** (French, International)
    
    Upload PDFs, ask questions via voice, and receive voice answers. Evaluate using BLEU, ROUGE, and human metrics.
    """)

    # Initialize session state
    if "listening" not in st.session_state:
        st.session_state["listening"] = False
    if "question_en" not in st.session_state:
        st.session_state["question_en"] = None

    # Model Descriptions
    st.subheader("Model Descriptions")
    for model_name, info in MODELS.items():
        st.markdown(f"**{model_name} ({info['language']})**: {info['description']}")

    # PDF Upload
    st.subheader("Upload PDF Documents")
    pdf_files = {
        "English": st.file_uploader("English PDF", type="pdf", key="en_pdf"),
        "Kannada": st.file_uploader("Kannada PDF", type="pdf", key="kn_pdf"),
        "French": st.file_uploader("French PDF", type="pdf", key="fr_pdf")
    }

    # Voice Input
    if st.button("Record Question", disabled=st.session_state["listening"]):
        question_en = speech_to_text()
        if question_en:
            question_kn = translate_text(question_en, "en", "kn")
            question_fr = translate_text(question_en, "en", "fr")
            st.session_state["questions"] = {
                "English": question_en,
                "Kannada": question_kn,
                "French": question_fr
            }

    # Process and Answer
    if "questions" in st.session_state and all(pdf_files.values()):
        st.subheader("Question-Answer Interactions")
        questions = st.session_state["questions"]
        auto_metrics = {"Model": [], "BLEU": [], "ROUGE-L": []}
        human_metrics = {"Model": [], "Fluency": [], "Coherence": [], "Relevance": [], "Average": []}
        comparison_table = {
            "Aspect": ["Language Fluency", "Named Entity Handling", "Accuracy of Extraction", "Speech-to-Text Accuracy", "Text-to-Speech Quality"],
            "FLAN-T5": ["", "", "", "", ""],
            "MuRIL": ["", "", "", "", ""],
            "mT5": ["", "", "", "", ""]
        }

        # Reference answers based on PDFs
        references = {
            "English": "Karnataka’s culture includes literature, music, dance, and cuisine, with Bengaluru as its capital.",
            "Kannada": "ಕರ್ನಾಟಕದ ಸಂಸ್ಕೃತಿಯು ಸಾಹಿತ್ಯ, ಸಂಗೀತ, ನೃತ್ಯ ಮತ್ತು ಆಹಾರವನ್ನು ಒಳಗೊಂಡಿದೆ, ಬೆಂಗಳೂರು ರಾಜಧಾನಿಯಾಗಿದೆ.",
            "French": "La culture française inclut la littérature, l'art et la cuisine, avec Paris comme capitale."
        }

        for lang, model_name in zip(["English", "Kannada", "French"], MODELS.keys()):
            pdf_file = pdf_files[lang]
            context = extract_text_from_pdf(pdf_file)
            if not context:
                continue
            question = questions[lang]
            
            with st.spinner(f"Processing {lang} with {model_name}..."):
                try:
                    answer = answer_question(model_name, context, question, lang)
                    st.markdown(f"### {lang} ({model_name})")
                    st.write(f"**Question**: {question}")
                    st.write(f"**Answer**: {answer}")
                    
                    # Voice Output
                    lang_code = "en" if lang == "English" else "kn" if lang == "Kannada" else "fr"
                    text_to_speech(answer, lang_code)
                    
                    # Evaluate
                    reference = references.get(lang, "Placeholder reference answer.")
                    metrics = evaluate_answer(reference, answer)
                    auto_metrics["Model"].append(model_name)
                    auto_metrics["BLEU"].append(metrics["bleu"])
                    auto_metrics["ROUGE-L"].append(metrics["rouge"]["rougeL"])
                    
                    st.write(f"**BLEU Score**: {metrics['bleu']:.4f}")
                    st.write(f"**ROUGE-L**: {metrics['rouge']['rougeL']:.4f}")
                    
                    # Human Evaluation
                    with st.expander(f"Rate {model_name} (Human Evaluation)"):
                        fluency = st.slider(f"{model_name} - Fluency", 1, 5, 3, key=f"{model_name}_f")
                        coherence = st.slider(f"{model_name} - Coherence", 1, 5, 3, key=f"{model_name}_c")
                        relevance = st.slider(f"{model_name} - Relevance", 1, 5, 3, key=f"{model_name}_r")
                        avg = (fluency + coherence + relevance) / 3
                        human_metrics["Model"].append(model_name)
                        human_metrics["Fluency"].append(fluency)
                        human_metrics["Coherence"].append(coherence)
                        human_metrics["Relevance"].append(relevance)
                        human_metrics["Average"].append(avg)
                    
                    # Update Comparison Table
                    comparison_table[model_name] = [
                        "✓✓✓" if fluency >= 4 else "✓✓",
                        "✓✓✓" if relevance >= 4 else "✓✓",
                        "High" if metrics["bleu"] > 0.5 else "Medium",
                        "✓✓✓" if question else "✓✓",
                        "Good" if answer and not answer.startswith("Erreur :") and not answer.startswith("Error:") else "Poor"
                    ]
                except Exception as e:
                    st.error(f"Error processing {model_name}: {e}")
                    logger.error(f"Error processing {model_name}: {e}")
                    continue

        # Visualization
        if auto_metrics["Model"]:
            st.subheader("📊 Comparative Results")
            df_auto = pd.DataFrame(auto_metrics).set_index("Model")
            st.markdown("### 🤖 Automatic Evaluation: BLEU & ROUGE-L")
            st.bar_chart(df_auto)

            df_human = pd.DataFrame(human_metrics).set_index("Model")
            st.markdown("### 👤 Human Evaluation: Fluency, Coherence, Relevance")
            st.bar_chart(df_human[["Fluency", "Coherence", "Relevance"]])

            st.markdown("### 🌟 Average Human Ratings")
            st.bar_chart(df_human[["Average"]])

            st.subheader("Comparison Table")
            df_comparison = pd.DataFrame(comparison_table).set_index("Aspect")
            st.table(df_comparison)

if __name__ == "__main__":
    main()
