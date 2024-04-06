from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# After training the model, save it
model.save_pretrained('./toxic_comment_model')