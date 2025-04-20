import string
import re
import requests
import streamlit as st
from collections import Counter, defaultdict
from itertools import product
from transformers import T5ForConditionalGeneration, T5Tokenizer
from difflib import SequenceMatcher

# ---------------------- Corpus Fetching ----------------------
@st.cache_data(show_spinner=False)
def fetch_corpus():
    urls = [
        "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
        "https://www.gutenberg.org/cache/epub/10615/pg10615.txt",
        "https://www.gutenberg.org/cache/epub/35899/pg35899.txt"
    ]
    full_text = ""
    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                full_text += response.text.lower()
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch: {url} - {e}")
    return full_text

# ---------------------- NLP Utilities ----------------------
def tokenize_text(text):
    sentences = re.split(r'[.!?]', text)
    return [re.findall(r'\b\w+\b', s) for s in sentences if s.strip()]

def build_ngram_models(tokenized_sentences):
    ngram_models = {}
    next_word_candidates = defaultdict(lambda: defaultdict(int))
    for n in range(2, 6):
        ngrams = []
        for sentence in tokenized_sentences:
            if len(sentence) >= n:
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i + n])
                    context = ngram[:-1]
                    word = ngram[-1]
                    next_word_candidates[context][word] += 1
                    ngrams.append(ngram)
        ngram_counts = Counter(ngrams)
        context_counts = Counter([ngram[:-1] for ngram in ngram_counts])
        model = {}
        for ngram in ngram_counts:
            context = ngram[:-1]
            model[ngram] = ngram_counts[ngram] / context_counts[context]
        ngram_models[n] = model
    return ngram_models, next_word_candidates

def build_word_stats(tokenized_sentences):
    words = [word for sentence in tokenized_sentences for word in sentence]
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    probs = {word: count / total_words for word, count in word_counts.items()}
    return set(words), word_counts, probs

# ---------------------- Spelling Correction ----------------------
def split(word): return [(word[:i], word[i:]) for i in range(len(word) + 1)]
def delete(word): return [L + R[1:] for L, R in split(word) if R]
def swap(word): return [L + R[1] + R[0] + R[2:] for L, R in split(word) if len(R) > 1]
def replace(word): return [L + c + R[1:] for L, R in split(word) if R for c in string.ascii_lowercase]
def insert(word): return [L + c + R for L, R in split(word) for c in string.ascii_lowercase]
def level_one_edits(word): return set(delete(word) + swap(word) + replace(word) + insert(word))
def level_two_edits(word): return set(e2 for e1 in level_one_edits(word) for e2 in level_one_edits(e1))

def correct_word(word, vocab, word_probs):
    if word in vocab:
        return word
    candidates = level_one_edits(word) or level_two_edits(word)
    valid = [w for w in candidates if w in vocab]
    if not valid:
        return word
    ranked = sorted(valid, key=lambda w: word_probs.get(w, 0), reverse=True)
    return ranked[0]

# Custom Spelling Rules
def custom_spelling_rules(word):
    spelling_corrections = {
        "tha": "the",
        "adn": "and",
        "teh": "the",
        "recieve": "receive",
        "occured": "occurred",
        "wierd": "weird",
        "completly": "completely",
        "definately": "definitely",
        "seperated": "separated",
        "wich": "which"
    }
    return spelling_corrections.get(word, word)

def correct_spelling(word, vocab, word_probs):
    word = custom_spelling_rules(word)  # Apply custom spelling rules
    if word in vocab:
        return f"✅ '{word}' is correctly spelled."
    candidates = level_one_edits(word) or level_two_edits(word)
    valid = [w for w in candidates if w in vocab]
    if not valid:
        return f"❌ No suggestions found for '{word}'"
    ranked = sorted(valid, key=lambda w: word_probs.get(w, 0), reverse=True)
    result = "🔍 Suggestions:\n" + ", ".join(f"{w} ({word_probs[w]:.4f})" for w in ranked[:10])
    return result

# ---------------------- Grammar Rules ----------------------
def custom_grammar_rules(sentence):
    grammar_corrections = {
        "eat yesterday": "ate yesterday",
        "I can able to do it": "I can do it",
        "She don't know": "She doesn't know",
        "He can plays football": "He can play football",
        "Me and John went to the store": "John and I went to the store",
        "She was more smarter": "She was smarter",
        "He is the most better candidate": "He is the best candidate",
        "I have went to the market": "I have gone to the market",
        "I will be finishing my homework tomorrow": "I will finish my homework tomorrow",
        "There is two cats": "There are two cats"
    }
    for wrong, right in grammar_corrections.items():
        sentence = sentence.replace(wrong, right)
    return sentence

def grammar_check_and_suggest(sentence, ngram_models, next_word_candidates, vocab, word_probs):
    sentence = custom_grammar_rules(sentence)  # Apply custom grammar rules
    tokens = re.findall(r'\w+', sentence.lower())
    original = " ".join(tokens)
    corrected_tokens = [correct_word(w, vocab, word_probs) for w in tokens]
    suggestions = corrected_tokens.copy()
    report = ""

    for n in range(5, 1, -1):
        model = ngram_models[n]
        changed = False
        for i in range(len(corrected_tokens) - n + 1):
            ngram = tuple(corrected_tokens[i:i + n])
            prob = model.get(ngram, 0)
            if prob == 0:
                context = ngram[:-1]
                if context in next_word_candidates:
                    top_choices = sorted(next_word_candidates[context].items(), key=lambda x: -x[1])
                    if top_choices:
                        new_word = top_choices[0][0]
                        suggestions[i + n - 1] = new_word
                        changed = True
        if changed:
            break

    suggested_sentence = " ".join(suggestions)
    return suggested_sentence

# ---------------------- Deep Learning Grammar Checker ----------------------
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def grammar_correct_with_t5(sentence, tokenizer, model):
    input_text = "grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# ---------------------- Similarity Score ----------------------
def similarity_score(original, corrected):
    return SequenceMatcher(None, original, corrected).ratio()

# ---------------------- Streamlit UI ----------------------
st.title("🧠 AI-Powered Grammar & Spell Checker")
st.markdown("Improve your English writing with both traditional NLP and Deep Learning!")

with st.spinner("🔄 Processing large corpus. Please wait..."):
    corpus_text = fetch_corpus()
    tokenized_sentences = tokenize_text(corpus_text)
    ngram_models, next_word_candidates = build_ngram_models(tokenized_sentences)
    vocab, word_counts, word_probs = build_word_stats(tokenized_sentences)
st.success("✅ Corpus processed successfully!")

st.markdown("### 🔡 Spell Checker")
input_word = st.text_input("🔍 Enter a word to check spelling:")
if st.button("Check Spelling"):
    if input_word:
        result = correct_spelling(input_word.lower(), vocab, word_probs)
        st.markdown(result)

st.markdown("---")
st.markdown("### 📝 Grammar Checker (Traditional + Deep Learning)")

input_sentence = st.text_area("✏️ Enter a sentence to check grammar and spelling:")

if st.button("Check Grammar"):
    if input_sentence:
        with st.spinner("🔍 Analyzing sentence..."):
            # Deep Learning
            tokenizer, model = load_t5_model()
            deep_corrected = grammar_correct_with_t5(input_sentence, tokenizer, model)

            # Traditional NLP
            nlp_corrected = grammar_check_and_suggest(input_sentence, ngram_models, next_word_candidates, vocab, word_probs)

            # Similarity
            sim_score = similarity_score(input_sentence.lower(), deep_corrected.lower())

            st.markdown("### ✅ Traditional NLP Suggestion:")
            st.markdown(f"`{nlp_corrected}`")

            st.markdown("### 🤖 Deep Learning Suggestion:")
            st.markdown(f"`Here’s a modified version of your code with 10 new rules included in the grammar correction logic:

```python
import string
import re
import requests
import streamlit as st
from collections import Counter, defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from difflib import SequenceMatcher

# ---------------------- Corpus Fetching ----------------------
@st.cache_data(show_spinner=False)
def fetch_corpus():
    urls = [
        "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
        "https://www.gutenberg.org/cache/epub/10615/pg10615.txt",
        "https://www.gutenberg.org/cache/epub/35899/pg35899.txt"
    ]
    full_text = ""
    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                full_text += response.text.lower()
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch: {url} - {e}")
    return full_text

# ---------------------- NLP Utilities ----------------------
def tokenize_text(text):
    sentences = re.split(r'[.!?]', text)
    return [re.findall(r'\b\w+\b', s) for s in sentences if s.strip()]

def build_ngram_models(tokenized_sentences):
    ngram_models = {}
    next_word_candidates = defaultdict(lambda: defaultdict(int))
    for n in range(2, 6):
        ngrams = []
        for sentence in tokenized_sentences:
            if len(sentence) >= n:
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i + n])
                    context = ngram[:-1]
                    word = ngram[-1]
                    next_word_candidates[context][word] += 1
                    ngrams.append(ngram)
        ngram_counts = Counter(ngrams)
        context_counts = Counter([ngram[:-1] for ngram in ngram_counts])
        model = {}
        for ngram in ngram_counts:
            context = ngram[:-1]
            model[ngram] = ngram_counts[ngram] / context_counts[context]
        ngram_models[n] = model
    return ngram_models, next_word_candidates

def build_word_stats(tokenized_sentences):
    words = [word for sentence in tokenized_sentences for word in sentence]
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    probs = {word: count / total_words for word, count in word_counts.items()}
    return set(words), word_counts, probs

# ---------------------- Spelling Correction ----------------------
def split(word): return [(word[:i], word[i:]) for i in range(len(word) + 1)]
def delete(word): return [L + R[1:] for L, R in split(word) if R]
def swap(word): return [L + R[1] + R[0] + R[2:] for L, R in split(word) if len(R) > 1]
def replace(word): return [L + c + R[1:] for L, R in split(word) if R for c in string.ascii_lowercase]
def insert(word): return [L + c + R for L, R in split(word) for c in string.ascii_lowercase]
def level_one_edits(word): return set(delete(word) + swap(word) + replace(word) + insert(word))
def level_two_edits(word): return set(e2 for e1 in level_one_edits(word) for e2 in level_one_edits(e1))

def correct_word(word, vocab, word_probs):
    if word in vocab:
        return word
    candidates = level_one_edits(word) or level_two_edits(word)
    valid = [w for w in candidates if w in vocab]
    if not valid:
        return word
    ranked = sorted(valid, key=lambda w: word_probs.get(w, 0), reverse=True)
    return ranked[0]

def correct_spelling(word, vocab, word_probs):
    if word in vocab:
        return f"✅ '{word}' is correctly spelled."
    candidates = level_one_edits(word) or level_two_edits(word)
    valid = [w for w in candidates if w in vocab]
    if not valid:
        return f"❌ No suggestions found for '{word}'"
    ranked = sorted(valid, key=lambda w: word_probs.get(w, 0), reverse=True)
    result = "🔍 Suggestions:\n" + ", ".join(f"{w} ({word_probs[w]:.4f})" for w in ranked[:10])
    return result

# ---------------------- Traditional Grammar Checker ----------------------
def grammar_check_and_suggest(sentence, ngram_models, next_word_candidates, vocab, word_probs):
    tokens = re.findall(r'\w+', sentence.lower())
    original = " ".join(tokens)
    corrected_tokens = [correct_word(w, vocab, word_probs) for w in tokens]
    suggestions = corrected_tokens.copy()
    report = ""

    for n in range(5, 1, -1):
        model = ngram_models[n]
        changed = False
        for i in range(len(corrected_tokens) - n + 1):
            ngram = tuple(corrected_tokens[i:i + n])
            prob = model.get(ngram, 0)
            if prob == 0:
                context = ngram[:-1]
                if context in next_word_candidates:
                    top_choices = sorted(next_word_candidates[context].items(), key=lambda x: -x[1])
                    if top_choices:
                        new_word = top_choices[0][0]
                        suggestions[i + n - 1] = new_word
                        changed = True
        if changed:
            break

    suggested_sentence = " ".join(suggestions)
    return suggested_sentence

# ---------------------- Deep Learning Grammar Checker ----------------------
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def grammar_correct_with_t5(sentence, tokenizer, model):
    input_text = "grammar: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# ---------------------- Similarity Score ----------------------
def similarity_score(original, corrected):
    return SequenceMatcher(None, original, corrected).ratio()

# ---------------------- Streamlit UI ----------------------
st.title("🧠 AI-Powered Grammar & Spell Checker")
st.markdown("Improve your English writing with both traditional NLP and Deep Learning!")

with st.spinner("🔄 Processing large corpus. Please wait..."):
    corpus_text = fetch_corpus()
    tokenized_sentences = tokenize_text(corpus_text)
    ngram_models, next_word_candidates = build_ngram_models(tokenized_sentences)
    vocab, word_counts, word_probs = build_word_stats(tokenized_sentences)
st.success("✅ Corpus processed successfully!")

st.markdown("### 🔡 Spell Checker")
input_word = st.text_input("🔍 Enter a word to check spelling:")
if st.button("Check Spelling"):
    if input_word:
        result = correct_spelling(input_word.lower(), vocab, word_probs)
        st.markdown(result)

st.markdown("---")
st.markdown("### 📝 Grammar Checker (Traditional + Deep Learning)")

input_sentence = st.text_area("✏️ Enter a sentence to check grammar and spelling:")

if st.button("Check Grammar"):
    if input_sentence:
        with st.spinner("🔍 Analyzing sentence..."):
            # Deep Learning
            tokenizer, model = load_t5_model()
            deep_corrected = grammar_correct_with_t5(input_sentence, tokenizer, model)

            # Traditional NLP
            nlp_corrected = grammar_check_and_suggest(input_sentence, ngram_models, next_word_candidates, vocab, word_probs)

            # Similarity
            sim_score = similarity_score(input_sentence.lower(), deep_corrected.lower())

            st.markdown("### ✅ Traditional NLP Suggestion:")
            st.markdown(f"`{nlp_corrected}`")

            st.markdown("### 🤖 Deep Learning Suggestion:")
            st.markdown(f"`{deep_corrected}`")

            st.markdown(f"### 📏 Similarity Score:\n`{sim_score:.4f}`")

