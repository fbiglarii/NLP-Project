import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# لود مدل
model_name = "kevinscaria/atsc_tk-instruct-base-def-pos-neg-neut-combined"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded!\n")

# ========================================
# ديتاست ها
# ========================================

datasets = {
    'Lapt14': 'Dataset/SemEval14/Test/Laptops_Test.csv',
    'Rest14': 'Dataset/SemEval14/Test/Restaurants_Test.csv',
    'Rest15': 'Dataset/SemEval15/Test/Restaurants_Test.csv',
    'Rest16': 'Dataset/SemEval16/Test/Restaurants_Test.csv',
}

# نتایج مقاله (InstructABSA-2)
paper_results = {
    'Lapt14': 81.56,
    'Rest14': 85.17,
    'Rest15': 84.50,
    'Rest16': 89.43,
}

# ========================================
# Prompt
# ========================================

instruction = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

# ========================================
# تابع ارزیابی 
# ========================================

def evaluate(dataset_path):
    df = pd.read_csv(dataset_path)
    
    correct = 0
    total = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['raw_text']
        aspects = eval(row['aspectTerms'])
        
        for asp in aspects:
            term = asp['term']
            true_polarity = asp['polarity'].lower()  
            

            if term == 'noaspectterm' or true_polarity == 'none':
                continue
            
            prompt = instruction.format(text=text, aspect=term)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=10)
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            if pred == true_polarity:
                correct += 1
            total += 1
    
    return correct / total * 100, total

# ========================================
# اجرا روی همه دیتاستها
# ========================================

our_results = {}

for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"⚠️ {name}: not found at {path}")
        continue
    
    print(f"\nEvaluating {name}...")
    acc, total = evaluate(path)
    our_results[name] = acc
    print(f"→ {name}: {acc:.2f}% ({total} samples)")

# ========================================
# جدول نهایی (مثل مقاله)
# ========================================

print(f"\n{'='*70}")
print(" Reproduction: ATSC Accuracy")
print(f"{'='*70}")
print(f"{'Model':<20} {'Lapt14':<12} {'Rest14':<12} {'Rest15':<12} {'Rest16':<12}")
print(f"{'-'*70}")

# نتایج مقاله
print(f"{'Paper (InstructABSA-2)':<20}", end="")
for name in ['Lapt14', 'Rest14', 'Rest15', 'Rest16']:
    print(f"{paper_results.get(name, '-'):<12.2f}", end="")
print()

# نتایج ما
print(f"{'Ours (Reproduced)':<20}", end="")
for name in ['Lapt14', 'Rest14', 'Rest15', 'Rest16']:
    if name in our_results:
        print(f"{our_results[name]:<12.2f}", end="")
    else:
        print(f"{'-':<12}", end="")
print()

print(f"{'='*70}")

# ذخیره CSV
results_df = pd.DataFrame({
    'Dataset': list(our_results.keys()),
    'Paper': [paper_results[k] for k in our_results.keys()],
    'Ours': list(our_results.values())
})
results_df.to_csv('Output/ATSC_reproduction.csv', index=False)
print("\n📁 Saved to Output/ATSC_reproduction.csv")
