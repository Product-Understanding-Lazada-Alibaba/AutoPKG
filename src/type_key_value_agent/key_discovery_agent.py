import os
import re
import csv
from transformers import pipeline

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_CSV = os.path.join(BASE_DIR, "..", "..", "data", "lazada_autopkg_kg_nodes.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "output", "program_output_temp")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "product_type_keys.csv")

# ─── Prompt template (type_key_value_prompts.py lines 26-48) ────────────
KEY_DISCOVERY_TEMPLATE = """Instruction: Generate a comprehensive table containing key product attributes.
The attributes must meet these criteria:
• Essential for both sellers and buyers
• Focused on inherent product characteristics (not logistics, packaging, or SEO)
• Based on industry standards or common e-commerce practices
• Includes visual and functional attributes that influence purchasing decisions
Table format:
• Attribute Name: Standard Attribute Name
• Description: What the attribute represents
• Examples: Examples of Standard Values (comprehensive)
Output constraints:
• Provide only the table, no extra text or explanation.
• Sort ALL attributes by importance from the buyer's perspective.
• Start with Brand as the top row, followed by other attributes in importance order.
• Use proper title case capitalization for all values in Attribute Name and Examples (e.g., Universal, Infrared,
Lithium-Ion, not universal, infrared, lithium-ion).
• Provide values in examples at least 5 if applicable.
• Keep example values less than 10.
Product Type: {product_type}
Product Type Description: {product_type_description}
"""


def load_product_types(csv_path: str) -> list[dict]:
    """Read nodes CSV and return all rows where node_type == 'Product Type'."""
    product_types = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("node_type") == "Product Type":
                product_types.append(row)
    return product_types


def build_prompt(row: dict) -> str:
    name = row.get("node_name", "")
    desc = row.get("description", "") or ""
    if desc.lower() == "null":
        desc = ""
    return KEY_DISCOVERY_TEMPLATE.format(
        product_type=name,
        product_type_description=desc,
    )


def clean_answer(raw: str) -> str:
    """Strip thinking tags and return the cleaned model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return cleaned


def main():
    # ── 1. Load model ────────────────────────────────────────────────────
    print("Loading Qwen3-235B-A22B model …")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-235B-A22B",
        device_map="auto",
        torch_dtype="auto",
    )
    print("Model loaded.\n")

    # ── 2. Read Product Type nodes ───────────────────────────────────────
    product_types = load_product_types(NODES_CSV)
    total = len(product_types)
    print(f"Found {total} Product Type nodes.\n")

    if total == 0:
        print("No Product Type nodes found. Exiting.")
        return

    # ── 3. Ensure output dir ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 4. Inference loop ────────────────────────────────────────────────
    results = []

    for idx, row in enumerate(product_types, start=1):
        node_id = row.get("node_id", "")
        type_name = row.get("node_name", "")
        prompt_text = build_prompt(row)
        messages = [{"role": "user", "content": prompt_text}]

        try:
            output = pipe(
                messages,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            generated = output[0]["generated_text"]
            # Chat models return list of message dicts
            if isinstance(generated, list):
                assistant_msgs = [m for m in generated if m.get("role") == "assistant"]
                raw_answer = assistant_msgs[-1]["content"] if assistant_msgs else ""
            else:
                # Plain text generation — strip prompt echo
                raw_answer = generated
                if prompt_text in raw_answer:
                    raw_answer = raw_answer[len(prompt_text):]

            keys_output = clean_answer(raw_answer)
        except Exception as e:
            print(f"  [ERROR] node_id={node_id}, type={type_name}: {e}")
            keys_output = "ERROR"

        results.append({
            "product_id": node_id,
            "product_type": type_name,
            "keys": keys_output,
        })
        print(f"[{idx}/{total}] {node_id} | {type_name}")
        print(f"  Keys preview: {keys_output[:120]}…\n")

    # ── 5. Save results ──────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "product_type", "keys"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
