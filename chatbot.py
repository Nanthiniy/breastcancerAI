from transformers import pipeline

# Load the pre-trained model
qa_pipeline = pipeline("question-answering", model="dmis-lab/biobert-base-cased", tokenizer="dmis-lab/biobert-base-cased")

def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
