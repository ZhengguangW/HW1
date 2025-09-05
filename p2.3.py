from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random

# Load pretrained DistilGPT2 model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

paragraph = (
    "The history of all hitherto existing society is the history of class struggles. "
    "Freeman and slave, patrician and plebeian, lord and serf, guild-master and journeyman, "
    "in a word, oppressor and oppressed, stood in constant opposition to one another. "
    "This struggle ended either in a revolutionary reconstitution of society at large or in the common ruin of the contending classes. "
    "In the earlier epochs of history, we find almost everywhere a complicated arrangement of society into various orders, "
    "a manifold gradation of social rank."
)

# Function to compute perplexity
def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

# Compute perplexity of original paragraph
ppl_original = compute_perplexity(paragraph)

# Shuffle the sentences randomly
sentences = paragraph.split(". ")
if sentences[-1].endswith("."):
    sentences[-1] = sentences[-1][:-1]  # remove trailing dot before shuffling
random.shuffle(sentences)
shuffled_paragraph = ". ".join(sentences) + "."

# Compute perplexity of shuffled paragraph
ppl_shuffled = compute_perplexity(shuffled_paragraph)

# Output results
print("=== Communist Manifesto Perplexity Analysis ===")
print(f"Original Paragraph:\n{paragraph}\n")
print(f"Shuffled Paragraph:\n{shuffled_paragraph}\n")
print(f"Original Perplexity: {ppl_original:.2f}")
print(f"Shuffled Perplexity: {ppl_shuffled:.2f}")


print("\n=== Sampling Comparison ===")

def generate_text(temp=None, use_greedy=False, max_tokens=500):
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if use_greedy:
        output = model.generate(
            input_ids,
            max_length=max_tokens,
            do_sample=False
        )
        strategy = "Greedy"
    else:
        output = model.generate(
            input_ids,
            max_length=max_tokens,
            do_sample=True,
            temperature=temp,
            top_k=0,  # use full vocabulary
            top_p=1.0
        )
        strategy = f"Temperature {temp}"

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- {strategy} ---\n{text[:800]}...\n")  

# (1) Greedy decoding
generate_text(use_greedy=True)

# (2) Temperature sampling
for T in [ 0.3, 0.6, 0.9, 1.2, 1.5]:
    generate_text(temp=T)