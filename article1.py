import streamlit as st
import spacy
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

# Load the Pegasus model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

# Load the SpaCy English model
nlp = spacy.load('en_core_web_sm')

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    st.title("Paraphrase Tool")
    user_input = st.text_area("Enter your paragraph:", "")
    
    if st.button("Paraphrase"):
        doc = nlp(user_input)
        sentences = list(doc.sents)

        paraphrased_sentences = []

        for sentence in sentences:
            paraphrased = get_paraphrased_sentences(model, tokenizer, sentence.text, num_beams=7, num_return_sequences=1)
            paraphrased_sentences.extend(paraphrased)

        paraphrased_text = ' '.join(paraphrased_sentences)
        
        st.subheader("Paraphrased Text:")
        st.write(paraphrased_text)

if __name__ == "__main__":
    main()
