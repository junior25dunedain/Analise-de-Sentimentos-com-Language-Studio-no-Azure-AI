from transformers import pipeline

# Criar pipelines para análise de sentimentos e emoções
sentiment_analysis = pipeline("sentiment-analysis")
emotion_analysis = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
def analisar_texto(texto):
    sentimento = sentiment_analysis(texto)[0]
    emocao = emotion_analysis(texto)[0]
    return {
        "Texto": texto,
        "Sentimento": sentimento['label'],
        "Score Sentimento": round(sentimento['score'], 2),
        "Emoção": emocao['label'],
        "Score Emoção": round(emocao['score'], 2)
    }

# Testar com exemplos
textos = [
    "Estou muito feliz com o produto!",
    "O atendimento foi péssimo."
]

resultados = [analisar_texto(texto) for texto in textos]

# Mostrar resultados
for resultado in resultados:
    print(resultado)
