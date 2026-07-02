import json

with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.loads(f.read(), strict=False)

# Format catalog as a minimal list of dicts
min_catalog = []
for item in catalog:
    min_catalog.append({
        "name": item["name"],
        "link": item["link"],
        "keys": item.get("keys", [])
    })

catalog_str = json.dumps(min_catalog)
char_count = len(catalog_str)
est_tokens = char_count / 4

print(f"Minimal catalog character count: {char_count}")
print(f"Estimated token count: {est_tokens:.2f} tokens")
print(f"Sample JSON item:\n{json.dumps(min_catalog[0], indent=2)}")
