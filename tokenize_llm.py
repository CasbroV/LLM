from transformers import GPT2Tokenizer

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Your input text
text=input("Enter text to tokenize: ")

# Tokenize the input
tokens=tokenizer.tokenize(text)
token_ids=tokenizer.convert_tokens_to_ids(tokens)

# Print results
print("\n----- Tokenization Results -----")
print("Original Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))

# Optional: Encode directly (tokens + ids)
encoded = tokenizer(text)
print("Encoded Output:", encoded)

# Decode back from IDs to string
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)