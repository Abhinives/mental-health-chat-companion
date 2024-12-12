from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load Blenderbot model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

# User input and FAISS response
user_input = "I feel worried"
faiss_response = "It's normal to feel worried. Try some deep breathing exercises to relax."

# Combine user input and FAISS response into a prompt with clear instructions
utterance = f"User is feeling worried and says: '{user_input}'. FAISS suggests the user: '{faiss_response}'. Act as a professional psychiatrist, and provide deeper empathetic advice, explore the user's feelings, and suggest additional coping strategies such as mindfulness, talking to a therapist, or self-care activities."

# Tokenize and generate response
inputs = tokenizer(utterance, return_tensors="pt")
res = model.generate(**inputs)

# Decode and print the output
print("Bot:", tokenizer.decode(res[0], skip_special_tokens=True))
