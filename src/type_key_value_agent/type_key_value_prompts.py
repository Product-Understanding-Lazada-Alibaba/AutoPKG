type_suggestion = """
Instruction: Return the simplest, most general product type that accurately represents the item while ensuring clarity and
avoiding ambiguity. Follow these rules:
• Remove brand names, model numbers, attributes, and marketing language.
• Eliminate redundant or situational descriptors.
• Standardize technical terms using widely accepted industry vocabulary.
• Abstract to the highest-level accurate category (e.g., "Chair" instead of "Office Chair").
• Use singular form, unless the product is commonly referred to in plural (e.g., "Scissors", "Pants", "Shoes").
• Return ‘None‘ for inputs that are unclear, ambiguous, or not valid products.
• If the input already meets all criteria, return it unchanged.
Additional Guidance:
• Contextual Clarity: Ensure the abstracted term maintains enough context to avoid ambiguity. For example, if the
original product is clearly related to vehicles, use terms like "Vehicle Part" instead of just "Part."
• Specificity Check: If the most general term could refer to multiple unrelated categories (e.g., "Panel" could mean a
house panel or a motorcycle panel), provide a more specific term that still fits the criteria (e.g., "Motorcycle Panel").
Your goal is to return the most common specific type that is still universally understood and accurately descriptive. Provide
only the product type in English, no extra text or explanation.

Title: xxx
Description: xxx
Specifications: xxx

"""


key_discovery = """
Instruction: Generate a comprehensive table containing key product attributes.
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
• Sort ALL attributes by importance from the buyer’s perspective.
• Start with Brand as the top row, followed by other attributes in importance order.
• Use proper title case capitalization for all values in Attribute Name and Examples (e.g., Universal, Infrared,
Lithium-Ion, not universal, infrared, lithium-ion).
• Provide values in examples at least 5 if applicable.
• Keep example values less than 10.
Product Type: xxx
Product Type Description: xxx

"""


value_extraction = """
Instruction: Given the table of attributes as a reference, extract relevant attributes from the provided input text and image.
Return a JSON object containing only the attribute ids and their corresponding attribute values.
• Extract JSON of attribute id and its corresponding attribute value.
• For attributes that have multiple values, return a list of values.
• If an attribute is not mentioned or cannot be determined with confidence, return null for that attribute.
• You are not restricted to the specific values in the example table — if another value makes sense based on the input,
feel free to use it.
• Cross-validate conflicting or ambiguous information between the text and image (e.g., if the text says “100% cotton”
but the image shows a shiny fabric, consider whether it might be polyester or a blend).
• Resolve ambiguities or duplicate values by selecting the most appropriate one based on context and consistency
(e.g., choosing between "L" and "Large" based on other entries or general conventions).
• Ensure the JSON output follows the structure and naming conventions shown in the example attribute table.
• Provide only the JSON, no extra text or explanation.
Simple Example:
• Input text: Brand: Philips. The machine body is made of ABS plastic and is available in White, Black, and Silver. It
has an IP44 rating for basic dust and splash protection.
• Output JSON: {"123": "Philips", "124": "IP44", "125": "ABS Plastic", "126": ["White", "Black", "Silver"]}

Product Type: xxx
Product Type Description: xxx
| Attribute ID | Attribute Name | Description | Examples |
|————–|—————-|————-|———-|
Title: xxx
Highlight: xxx
Description: xxx
Specifications: xxx
Images: xxx 
"""