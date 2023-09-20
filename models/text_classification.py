import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict_zero_shot(text, label_texts, label='entailment', normalize=True):
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    tokens = tokenizer([text] * len(label_texts), label_texts,
                       truncation=True, return_tensors='pt', padding=True)
    with torch.inference_mode():
        result = torch.softmax(model(**tokens.to(model.device)).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return proba
