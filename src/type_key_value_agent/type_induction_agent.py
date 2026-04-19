import os
import csv
import sys
from transformers import pipeline

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "..", "..", "data", "lazada_autopkg_product_data_backup.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "output", "program_output_temp")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "product_type_results.csv")

# ─── Prompt template (type_suggestion lines 1-23) ───────────────────────
TYPE_SUGGESTION_TEMPLATE = """Instruction: Return the simplest, most general product type that accurately represents the item while ensuring clarity and
avoiding ambiguity. Follow these rules:
• Remove brand names, model numbers, attributes, and marketing language.
• Eliminate redundant or situational descriptors.
• Standardize technical terms using widely accepted industry vocabulary.
• Abstract to the highest-level accurate category (e.g., "Chair" instead of "Office Chair").
• Use singular form, unless the product is commonly referred to in plural (e.g., "Scissors", "Pants", "Shoes").
• Return 'None' for inputs that are unclear, ambiguous, or not valid products.
• If the input already meets all criteria, return it unchanged.
Additional Guidance:
• Contextual Clarity: Ensure the abstracted term maintains enough context to avoid ambiguity. For example, if the
original product is clearly related to vehicles, use terms like "Vehicle Part" instead of just "Part."
• Specificity Check: If the most general term could refer to multiple unrelated categories (e.g., "Panel" could mean a
house panel or a motorcycle panel), provide a more specific term that still fits the criteria (e.g., "Motorcycle Panel").
Your goal is to return the most common specific type that is still universally understood and accurately descriptive. Provide
only the product type in English, no extra text or explanation.

Title: {title}
Description: {description}
Specifications: {specifications}
"""


def build_prompt(row: dict) -> str:
    title = row.get("product_name", "") or ""
    description = row.get("description", "") or ""
    specifications = row.get("specifications", "") or ""
    # treat literal "null" as empty
    if title.lower() == "null":
        title = ""
    if description.lower() == "null":
        description = ""
    if specifications.lower() == "null":
        specifications = ""
    return TYPE_SUGGESTION_TEMPLATE.format(
        title=title,
        description=description,
        specifications=specifications,
    )


def extract_type(generated_text: str, prompt: str) -> str:
    """Extract the product type from the generated text."""
    # Remove the prompt portion if the model echoes it
    answer = generated_text
    if prompt in answer:
        answer = answer[len(prompt):]
    # Take first non-empty line as the type
    for line in answer.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return "None"


def main():
    # ── 1. Load model ────────────────────────────────────────────────────
    print("Loading Qwen3-4B-Instruct model …")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-4B-Instruct-2507",
        device_map="auto",
        torch_dtype="auto",
    )
    print("Model loaded.\n")

    # ── 2. Read CSV ──────────────────────────────────────────────────────
    products = []
    with open(DATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    total = len(products)
    print(f"Read {total} products from CSV.\n")

    # ── 3. Ensure output dir ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 4. Inference loop ────────────────────────────────────────────────
    results = []
    BATCH_SAVE_INTERVAL = 50  # save progress every N products

    for idx, row in enumerate(products, start=1):
        product_id = row.get("product_id", "")
        prompt_text = build_prompt(row)
        messages = [{"role": "user", "content": prompt_text}]

        try:
            output = pipe(
                messages,
                max_new_tokens=30,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            generated = output[0]["generated_text"]
            # generated is a list of message dicts for chat models
            if isinstance(generated, list):
                # take the last assistant message
                assistant_msgs = [m for m in generated if m.get("role") == "assistant"]
                raw_answer = assistant_msgs[-1]["content"] if assistant_msgs else ""
            else:
                raw_answer = extract_type(generated, prompt_text)

            # Clean: take first line, strip thinking tags if present
            product_type = raw_answer.strip()
            # Remove <think>...</think> blocks (Qwen3 thinking mode)
            import re
            product_type = re.sub(r"<think>.*?</think>", "", product_type, flags=re.DOTALL).strip()
            # Take first non-empty line
            for line in product_type.splitlines():
                line = line.strip()
                if line:
                    product_type = line
                    break
            else:
                product_type = "None"

        except Exception as e:
            print(f"  [ERROR] product_id={product_id}: {e}")
            product_type = "None"

        results.append({"product_id": product_id, "type": product_type})
        print(f"[{idx}/{total}] {product_id} -> {product_type}")

        # Periodic save
        if idx % BATCH_SAVE_INTERVAL == 0:
            _save_csv(results)
            print(f"  (progress saved: {idx}/{total})")

    # ── 5. Final save ────────────────────────────────────────────────────
    _save_csv(results)
    print(f"\nDone! Results saved to {OUTPUT_CSV}")


def _save_csv(results: list):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "type"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
