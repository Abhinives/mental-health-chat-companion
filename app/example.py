from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Input text
input_text = "hello i am abhinivesh"
max_length = 128

# Encode input text with attention mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Generate response
response = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # Pass attention mask
    max_length=max_length,
    num_beams=5,
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set the pad token ID
    do_sample=False,
    top_k=50,        # Use top-k sampling for diverse responses
    top_p=0.95 
)

# Decode and print the response
generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
print(generated_text)
