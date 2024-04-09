from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
import torch

# Carregar o tokenizador e o modelo pré-treinado do GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForQuestionAnswering.from_pretrained("gpt2")

# Texto de contexto
contexto = """
O GPT-2 (Generative Pre-trained Transformer 2) é um modelo de linguagem de inteligência artificial 
desenvolvido pela OpenAI. Ele é treinado em uma tarefa de previsão de palavras em uma grande quantidade 
de texto da internet para gerar texto de qualidade quando alimentado com uma entrada de texto. O GPT-2 
é um dos modelos mais avançados de linguagem atualmente disponíveis e é capaz de gerar texto altamente 
coerente e semelhante ao texto humano em uma variedade de tarefas.
"""

# Perguntas
pergunta1 = "O que é o GPT-2?"
pergunta2 = "Quem desenvolveu o GPT-2?"

# Tokenizar o texto de contexto e as perguntas
inputs1 = tokenizer(contexto, pergunta1, add_special_tokens=True, return_tensors="pt")
inputs2 = tokenizer(contexto, pergunta2, add_special_tokens=True, return_tensors="pt")

# Obter a saída do modelo para responder à primeira pergunta
outputs1 = model(**inputs1)

# Obter a saída do modelo para responder à segunda pergunta
outputs2 = model(**inputs2)

# Decodificar as respostas
start_index1 = torch.argmax(outputs1.start_logits)
end_index1 = torch.argmax(outputs1.end_logits)
resposta1 = tokenizer.decode(inputs1["input_ids"].squeeze()[start_index1:end_index1+1])

start_index2 = torch.argmax(outputs2.start_logits)
end_index2 = torch.argmax(outputs2.end_logits)
resposta2 = tokenizer.decode(inputs2["input_ids"].squeeze()[start_index2:end_index2+1])

print("Resposta à pergunta 1:", resposta1)
print("Resposta à pergunta 2:", resposta2)
