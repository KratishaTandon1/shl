import json
import re

with open("shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.loads(f.read(), strict=False)

with open("extracted_recommendations.json", "r", encoding="utf-8") as f:
    extracted_recs = json.load(f)

# Group extracted recommendations by trace file
trace_gt = {}
for item in extracted_recs:
    for src in item["sources"]:
        if src not in trace_gt:
            trace_gt[src] = []
        trace_gt[src].append(item["name"])

# Load traces conversation history
def load_trace_history(filename):
    path = f"c:\\Users\\prakh\\Downloads\\shl\\{filename}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    messages = []
    turns = content.split("### Turn ")
    for turn in turns[1:]:
        turn_lines = turn.strip().split("\n")
        user_text = ""
        agent_text = ""
        
        user_start = -1
        agent_start = -1
        for i, line in enumerate(turn_lines):
            if "**User**" in line:
                user_start = i + 1
            elif "**Agent**" in line:
                agent_start = i + 1
                
        if user_start != -1:
            end_idx = agent_start - 1 if agent_start != -1 else len(turn_lines)
            user_lines = [l.strip("> ").strip() for l in turn_lines[user_start:end_idx] if l.strip()]
            user_text = " ".join(user_lines)
            messages.append({"role": "user", "content": user_text})
            
        if agent_start != -1:
            agent_lines = []
            for l in turn_lines[agent_start:]:
                l_strip = l.strip()
                if l_strip.startswith("|") or l_strip.startswith("_") or l_strip.startswith("###"):
                    break
                if l_strip:
                    agent_lines.append(l_strip)
            agent_text = " ".join(agent_lines)
            messages.append({"role": "assistant", "content": agent_text})
            
    return messages

def retrieve_relevant_products(messages, catalog):
    import agent
    return agent.retrieve_relevant_products(messages, catalog)

print("Verifying retrieval on each trace:")
all_passed = True
for trace_file, gt_names in trace_gt.items():
    history = load_trace_history(trace_file)
    retrieved = retrieve_relevant_products(history, catalog)
    import catalog as cat_lib
    retrieved_names = {item["name"] for item in retrieved}
    retrieved_urls = {item["link"] for item in retrieved}
    
    missing = []
    for name in gt_names:
        res = cat_lib.resolve_recommendation(name, "")
        if res:
            res_name, res_url, _ = res
            if res_url not in retrieved_urls:
                missing.append(name)
        else:
            if name not in retrieved_names:
                missing.append(name)
                
    if missing:
        print(f"  {trace_file}: FAILED. Missing recommendations: {missing}")
        all_passed = False
    else:
        print(f"  {trace_file}: PASSED. All {len(gt_names)} recommendations retrieved in top set.")

if all_passed:
    print("\nSUCCESS! The retrieval strategy covers 100% of the ground truth recommendations across all traces.")
else:
    print("\nSome recommendations were missed. We need to tune the retrieval logic.")
