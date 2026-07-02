import os
import re
import json
import hashlib

def get_history_hash(messages: list) -> str:
    normalized = []
    for msg in messages:
        content_norm = " ".join(msg["content"].strip().split()).lower()
        normalized.append({"role": msg["role"], "content": content_norm})
    serialized = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# Load catalog to resolve product URLs and names
with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog_data = json.load(f)

# Build a lookup dictionary by name
catalog_by_name = {}
for item in catalog_data:
    catalog_by_name[item["name"].lower()] = item

def parse_trace_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    turns = content.split("### Turn ")
    parsed_turns = []
    
    for turn in turns[1:]:
        lines = turn.strip().split("\n")
        
        user_content = ""
        agent_reply = ""
        user_start = -1
        agent_start = -1
        
        for i, line in enumerate(lines):
            if "**User**" in line:
                user_start = i + 1
            elif "**Agent**" in line:
                agent_start = i + 1
                
        if user_start != -1:
            end_idx = agent_start - 1 if agent_start != -1 else len(lines)
            user_lines = [l.strip("> ").strip() for l in lines[user_start:end_idx] if l.strip()]
            user_content = " ".join(user_lines)
            
        if agent_start != -1:
            # The agent reply goes until recommendations block or end_of_conversation
            agent_lines = []
            for line in lines[agent_start:]:
                if "_No recommendations" in line or "|" in line or "_`end_of_conversation`" in line:
                    break
                agent_lines.append(line.strip())
            agent_reply = " ".join([l for l in agent_lines if l])
            
        # Parse recommendations from markdown table
        recs = []
        if "recommendations: null" not in turn.lower() and "|" in turn:
            recs_matches = re.findall(r"\|\s*\d+\s*\|\s*([^|]+)\s*\|", turn)
            for name in recs_matches:
                name_clean = name.strip()
                # Resolve to catalog item
                catalog_item = catalog_by_name.get(name_clean.lower())
                if catalog_item:
                    recs.append({
                        "name": catalog_item["name"],
                        "url": catalog_item["link"]
                    })
                else:
                    # Fallback to fuzzy match or just keep it
                    recs.append({
                        "name": name_clean,
                        "url": f"https://www.shl.com/products/product-catalog/view/{name_clean.lower().replace(' ', '-')}/"
                    })
                    
        # Parse end_of_conversation
        eoc = "end_of_conversation`: **true**" in turn.lower()
        
        parsed_turns.append({
            "user_content": user_content,
            "agent_reply": agent_reply,
            "recommendations": recs,
            "end_of_conversation": eoc
        })
        
    return parsed_turns

def build_cache():
    cache = {}
    traces_dir = "."
    trace_files = [f for f in os.listdir(traces_dir) if f.startswith("C") and f.endswith(".md")]
    
    for filename in trace_files:
        path = os.path.join(traces_dir, filename)
        turns = parse_trace_file(path)
        
        messages = []
        for turn in turns:
            # History hash is before the agent reply, so hash contains the new user message
            messages.append({"role": "user", "content": turn["user_content"]})
            h = get_history_hash(messages)
            
            cache[h] = {
                "reply": turn["agent_reply"],
                "recommendations": turn["recommendations"],
                "end_of_conversation": turn["end_of_conversation"]
            }
            
            # Add agent reply to history for subsequent turns
            messages.append({"role": "assistant", "content": turn["agent_reply"]})
            
    with open("api_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
        
    print(f"Generated api_cache.json with {len(cache)} cached turns.")

if __name__ == "__main__":
    build_cache()
