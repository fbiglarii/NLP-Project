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
print("Model loaded!\n")

# ========================================
# داده تست فارسی
# ========================================

persian_test = [
    {"text": "غذا عالی بود", "aspect": "غذا", "polarity": "positive"},
    {"text": "پیتزا خوشمزه بود", "aspect": "پیتزا", "polarity": "positive"},
    {"text": "سرویس افتضاح بود", "aspect": "سرویس", "polarity": "negative"},
    {"text": "گارسون بی ادب بود", "aspect": "گارسون", "polarity": "negative"},
    {"text": "قیمت معمولی بود", "aspect": "قیمت", "polarity": "neutral"},
    {"text": "رستوران شلوغ بود", "aspect": "رستوران", "polarity": "neutral"},
    {"text": "کیفیت غذا بد بود", "aspect": "کیفیت غذا", "polarity": "negative"},
    {"text": "فضای رستوران دلنشین بود", "aspect": "فضا", "polarity": "positive"},
    {"text": "منو متنوع بود", "aspect": "منو", "polarity": "positive"},
    {"text": "انتظار زیادی کشیدیم", "aspect": "انتظار", "polarity": "negative"},
    {"text": "پارکینگ داشت", "aspect": "پارکینگ", "polarity": "neutral"},
    {"text": "دسر عالی بود", "aspect": "دسر", "polarity": "positive"},
    {"text": "نوشیدنی سرد نبود", "aspect": "نوشیدنی", "polarity": "negative"},
    {"text": "میز تمیز بود", "aspect": "میز", "polarity": "positive"},
    {"text": "صندلی راحت نبود", "aspect": "صندلی", "polarity": "negative"},
]

print(f"Persian test samples: {len(persian_test)}")

# ========================================
# Promptها
# ========================================

# Zero-shot (بدون مثال)
prompt_zero_shot = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

# 6-shot فارسی
prompt_6_shot_fa = """Definition: The output will be 'positive', 'negative', or 'neutral' based on the sentiment of the aspect.

Example 1-
input: غذا خوشمزه بود. The aspect is غذا.
output: positive

Example 2-
input: فضای رستوران عالی بود. The aspect is فضا.
output: positive

Example 3-
input: سرویس بد بود. The aspect is سرویس.
output: negative

Example 4-
input: زمان انتظار طولانی بود. The aspect is انتظار.
output: negative

Example 5-
input: قیمت مناسب بود. The aspect is قیمت.
output: neutral

Example 6-
input: رستوران در مرکز شهر بود. The aspect is رستوران.
output: neutral

Now complete the following example-
input: {text} The aspect is {aspect}.
output:"""

prompts = {
    'Zero-shot': prompt_zero_shot,
    '6-shot Persian': prompt_6_shot_fa,
}

# ========================================
# تابع ارزیابی
# ========================================

def evaluate(prompt_template):
    correct = 0
    total = 0
    
    for item in persian_test:
        text = item['text']
        aspect = item['aspect']
        true_polarity = item['polarity'].lower()
        
        prompt = prompt_template.format(text=text, aspect=aspect)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        if pred == true_polarity:
            correct += 1
        total += 1
        
        # نمایش جزئیات
        status = "✅" if pred == true_polarity else "❌"
        print(f"{status} {text:<25} | True: {true_polarity:<10} Pred: {pred}")
    
    return correct / total * 100, correct, total

# ========================================
# اجرا
# ========================================

results = {}

for name, prompt in prompts.items():
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    acc, correct, total = evaluate(prompt)
    results[name] = acc
    
    print(f"\n→ Correct: {correct}/{total}")
    print(f"→ Accuracy: {acc:.2f}%")

# ========================================
# نتیجه نهایی
# ========================================

print(f"\n{'='*60}")
print("📊 Final Results - Persian ATSC")
print(f"{'='*60}")
print(f"{'Prompt':<20} {'Accuracy':<15}")
print(f"{'-'*60}")

for name, acc in results.items():
    bar = "█" * int(acc / 5)
    print(f"{name:<20} {acc:.2f}%  {bar}")

# مقایسه
diff = results['6-shot Persian'] - results['Zero-shot']
sign = "+" if diff > 0 else ""
print(f"\n🔍 Improvement: {sign}{diff:.2f}%")

if diff > 0:
    print("✅ Few-shot Was Effective !")
else:
    print("⚠️ Few-shot Was Not Effective")

# ذخیره
results_df = pd.DataFrame({
    'Prompt': list(results.keys()),
    'Accuracy': list(results.values())
})
results_df.to_csv('Output/persian_zero_vs_6shot.csv', index=False)
print("\n📁 Saved to Output/persian_zero_vs_6shot.csv")
