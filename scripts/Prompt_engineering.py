import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# لود مدل
model_name = "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded!")

# ========================================
# Promptها: 4 و 8 مثال
# ========================================

prompt_4_examples = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Example 1-
input: The food was delicious. The aspect is food.
output: positive

Example 2-
input: The service was terrible. The aspect is service.
output: negative

Example 3-
input: The price is reasonable. The aspect is price.
output: neutral

Example 4-
input: I loved the atmosphere. The aspect is atmosphere.
output: positive

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

prompt_8_examples = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Example 1-
input: The food was delicious. The aspect is food.
output: positive

Example 2-
input: I loved the atmosphere. The aspect is atmosphere.
output: positive

Example 3-
input: Great service and friendly staff. The aspect is service.
output: positive

Example 4-
input: The service was terrible. The aspect is service.
output: negative

Example 5-
input: The wait time was too long. The aspect is wait time.
output: negative

Example 6-
input: Overpriced for what you get. The aspect is price.
output: negative

Example 7-
input: The restaurant is located downtown. The aspect is restaurant.
output: neutral

Example 8-
input: They serve Italian food. The aspect is food.
output: neutral

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

prompts = {
    '4 Examples': prompt_4_examples,
    '8 Examples': prompt_8_examples,
}

# ========================================
# لود دیتاست (کل Rest15)
# ========================================

df = pd.read_csv('Dataset/SemEval15/Test/Restaurants_Test.csv')

samples = []
for _, row in df.iterrows():
    text = row['raw_text']
    aspects = eval(row['aspectTerms'])
    for asp in aspects:
        term = asp['term']
        polarity = asp['polarity'].lower()
        if term != 'noaspectterm' and polarity != 'none':
            samples.append({
                'text': text,
                'aspect': term,
                'polarity': polarity
            })

print(f"Testing on ALL {len(samples)} samples")

# ========================================
# تابع ارزیابی
# ========================================

def evaluate(prompt_template):
    correct = 0
    
    for s in tqdm(samples):
        prompt = prompt_template.format(text=s['text'], aspect=s['aspect'])
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        if pred == s['polarity']:
            correct += 1
    
    return correct / len(samples) * 100

# ========================================
# اجرا
# ========================================

results = {}
for prompt_name, prompt_template in prompts.items():
    print(f"\n{'='*50}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*50}")
    acc = evaluate(prompt_template)
    results[prompt_name] = acc
    print(f"→ Accuracy: {acc:.2f}%")

# ========================================
# نتایج
# ========================================

print(f"\n{'='*60}")
print(f"RESULTS - InstructABSA on Rest15 ({len(samples)} samples)")
print(f"{'='*60}")
print(f"{'Prompt':<20} {'Accuracy':<15} {'vs Paper'}")
print(f"{'-'*60}")
print(f"{'Paper (6 examples)':<20} {'84.50%':<15} {'-'}")

for name, acc in results.items():
    diff = acc - 84.50
    sign = "+" if diff > 0 else ""
    print(f"{name:<20} {acc:.2f}%{'':<10} {sign}{diff:.2f}%")

print(f"{'='*60}")

# بهترین
best = max(results, key=results.get)
print(f"\n🏆 Best: {best} ({results[best]:.2f}%)")

# ذخیره
results_df = pd.DataFrame({
    'Prompt': ['Paper (6 examples)'] + list(results.keys()),
    'Accuracy': [84.50] + list(results.values())
})
results_df.to_csv('Output/prompt_4_8_comparison.csv', index=False)
print("\n📁 Saved to Output/prompt_4_8_comparison.csv")
