# 🤖 Extractive Question Answering with BERT

This repository implements an extractive Question Answering (QA) pipeline using a pre-trained BERT model fine-tuned on the SQuAD (Stanford Question Answering Dataset).

## 🚀 Overview

Given a reference passage (context) and a question, the model locates the span of text within the passage that answers the question. It achieves this by calculating start and end token probability scores.

## 🧠 Pipeline Architecture

1. **Tokenization:** Encodes the question and passage together using `BertTokenizer.encode_plus`, applying truncation and generating PyTorch tensors.
2. **Attention Masking:** Identifies valid tokens and pads inputs cleanly.
3. **Span Prediction:** Feeds representations through `BertForQuestionAnswering` without calculating gradients (`torch.no_grad()`).
4. **Index Selection:** Extracts the most probable starting and ending token positions using `torch.argmax`.
5. **Decoding:** Converts selected token IDs back into human-readable text using `convert_tokens_to_string`.

## 🛠️ Tech Stack
- Python
- PyTorch
- Hugging Face Transformers (`bert-large-uncased-whole-word-masking-finetuned-squad`)

## 💻 Installation & Usage

1. Install required packages:
pip install torch transformers

2. Run the script:
python qa_bert.py

## 📊 Example

- **Context:** "France, officially the French Republic, is a country whose capital is Paris"
- **Question:** "What is the capital of France?"
- **Answer:** "paris"
