# ============================================
# Advanced NLP Project: DistilBERT + GPT-2
# ============================================

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# 1. Load Dataset (IMDB)
# =========================
dataset = load_dataset("imdb")
train_ds = dataset["train"].shuffle(seed=42).select(range(2000))
test_ds = dataset["test"].shuffle(seed=42).select(range(1000))

# =========================
# 2. Fine-tune DistilBERT
# =========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 2
model.train()
loss_values = []

for epoch in range(num_epochs):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# =========================
# 3. Evaluation
# =========================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
acc = accuracy_score(all_labels, all_preds)

print("\n=== DistilBERT Classification Performance ===")
print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Visualization
plt.plot(loss_values)
plt.title("Training Loss Over Steps")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()

# =========================
# 4. Inference (Classification)
# =========================
def predict_sentiment(texts):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    return preds.cpu().numpy()

examples = [
    "I absolutely loved this movie! The story was great.",
    "This film was boring and way too long.",
]

print("\n=== DistilBERT Predictions ===")
for text, pred in zip(examples, predict_sentiment(examples)):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Text: {text}\nPrediction: {label}\n")

# =========================
# 5. GPT-2 for Generation
# =========================
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_pipeline = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer, max_length=100)

print("\n=== GPT-2 Generations ===")
prompts = [
    "Artificial Intelligence will change the future because",
    "The movie was so emotional that"
]

for p in prompts:
    output = gen_pipeline(p, num_return_sequences=1)[0]["generated_text"]
    print(f"Prompt: {p}\nGenerated: {output}\n")

# =========================
# 6. LangChain Integration
# =========================
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = HuggingFacePipeline(pipeline=gen_pipeline)
prompt = PromptTemplate(template="Write a creative reply to: {question}", input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

print("\n=== LangChain Integration ===")
print(chain.run("What is the future of AI in education?"))

# =========================
# 7. Research Questions (printed in notebook/logs)
# =========================
print("\n=== Research Questions ===")
print("1. How well does DistilBERT capture sentiment context vs. GPT-2 in free text?")
print("2. Does GPT-2 show creativity but lack factual consistency?")
print("3. How does fine-tuning impact classification accuracy vs. using pre-trained only?")
print("4. What are the trade-offs between interpretability and performance?")
