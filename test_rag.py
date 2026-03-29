from src.pipeline import generate_answer

query = "What is machine learning?"

response = generate_answer(query)

print("\n🧠 Final Answer:\n")
print(response)