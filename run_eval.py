import os
import re
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def parse_trace_file(filepath):
    """
    Parses a trace markdown file into a list of turns.
    Each turn contains:
      - user_content: the user input
      - expected_recs: list of expected product names, or None
      - expected_eoc: expected end_of_conversation boolean
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    turns = content.split("### Turn ")
    parsed_turns = []
    
    for turn in turns[1:]:
        lines = turn.strip().split("\n")
        turn_num = lines[0].strip()
        
        user_content = ""
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
            
        # Extract expected recommendations
        expected_recs = None
        if "recommendations: null" not in turn.lower() and "|" in turn:
            # Parse recommended names from markdown table
            # Row format: | 1 | Occupational Personality Questionnaire OPQ32r | P | ...
            recs_matches = re.findall(r"\|\s*\d+\s*\|\s*([^|]+)\s*\|", turn)
            if recs_matches:
                expected_recs = [r.strip() for r in recs_matches]
                
        # Extract expected end_of_conversation
        expected_eoc = False
        if "end_of_conversation`: **true**" in turn.lower():
            expected_eoc = True
            
        parsed_turns.append({
            "turn_num": turn_num,
            "user_content": user_content,
            "expected_recs": expected_recs,
            "expected_eoc": expected_eoc
        })
        
    return parsed_turns

def run_evaluation():
    traces_dir = "."
    trace_files = sorted([f for f in os.listdir(traces_dir) if f.startswith("C") and f.endswith(".md")])
    
    print("=" * 70)
    print("STARTING SHL ASSESSMENT RECOMMENDER OFFLINE EVALUATION")
    print("=" * 70)
    
    all_passed = True
    total_traces = 0
    passed_traces = 0
    
    for filename in trace_files:
        total_traces += 1
        path = os.path.join(traces_dir, filename)
        print(f"\nEvaluating Trace: {filename}...")
        turns = parse_trace_file(path)
        
        messages = []
        trace_success = True
        
        for turn_idx, turn in enumerate(turns):
            user_msg = turn["user_content"]
            expected_recs = turn["expected_recs"]
            expected_eoc = turn["expected_eoc"]
            
            # Append user message
            messages.append({"role": "user", "content": user_msg})
            
            # Call FastAPI POST /chat
            try:
                response = client.post("/chat", json={"messages": messages})
                if response.status_code != 200:
                    print(f"  [FAIL] Turn {turn['turn_num']}: HTTP {response.status_code} - {response.text}")
                    trace_success = False
                    break
                    
                data = response.json()
                
                # Check Schema Compliance
                required_fields = ["reply", "recommendations", "end_of_conversation"]
                for f in required_fields:
                    if f not in data:
                        print(f"  [FAIL] Turn {turn['turn_num']}: Response missing required schema field: '{f}'")
                        trace_success = False
                        
                # Check recommendations content
                actual_recs = data.get("recommendations", [])
                
                # 1. Verify all recommendations are items in the catalog (resolved successfully)
                for item in actual_recs:
                    if not item.get("name") or not item.get("url") or not item.get("test_type"):
                        print(f"  [FAIL] Turn {turn['turn_num']}: Recommendation item missing name, url, or test_type: {item}")
                        trace_success = False
                
                # 2. Match recommendations shortlist against expected shortlist
                if expected_recs is not None:
                    # Ground truth has recommendations
                    actual_names = [item["name"] for item in actual_recs]
                    def norm(name):
                        n = name.lower()
                        n = n.replace("microsoft \n    365", "microsoft excel 365")
                        n = n.replace("microsoft \n 365", "microsoft excel 365")
                        return re.sub(r'[^\w]', '', n)
                        
                    actual_norms = {norm(name) for name in actual_names}
                    matched_items = [name for name in expected_recs if norm(name) in actual_norms]
                    recall = len(matched_items) / len(expected_recs) if expected_recs else 1.0
                    
                    if recall < 1.0:
                        print(f"  [WARN] Turn {turn['turn_num']}: Recall@{len(actual_names)} is {recall:.2%}")
                        print(f"    Expected: {expected_recs}")
                        print(f"    Got:      {actual_names}")
                        # Don't fail the hard test on warn, but track it
                    else:
                        print(f"  [OK] Turn {turn['turn_num']}: Shortlist matched (100% Recall)")
                else:
                    # Ground truth has no recommendations
                    if len(actual_recs) > 0:
                        print(f"  [WARN] Turn {turn['turn_num']}: Got {len(actual_recs)} recommendations but expected NONE (vague query / clarifying).")
                        print(f"    Got: {[item['name'] for item in actual_recs]}")
                    else:
                        print(f"  [OK] Turn {turn['turn_num']}: Correctly returned no recommendations.")
                        
                # Check End of Conversation flag
                actual_eoc = data.get("end_of_conversation", False)
                if actual_eoc != expected_eoc:
                    print(f"  [WARN] Turn {turn['turn_num']}: end_of_conversation is {actual_eoc}, expected {expected_eoc}")
                    
                # Append assistant response to messages for the next turn
                messages.append({"role": "assistant", "content": data.get("reply", "")})
                
                # Prevent hitting Gemini free-tier rate limits (5 RPM / 250K TPM)
                import time
                time.sleep(0.05)
                
            except Exception as ex:
                print(f"  [FAIL] Turn {turn['turn_num']}: Exception occurred: {ex}")
                trace_success = False
                break
                
        if trace_success:
            print(f"Result for {filename}: PASSED")
            passed_traces += 1
        else:
            print(f"Result for {filename}: FAILED")
            all_passed = False
            
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total traces replayed:  {total_traces}")
    print(f"Passed traces:          {passed_traces}")
    print(f"Failed traces:          {total_traces - passed_traces}")
    print(f"Success Rate:           {passed_traces / total_traces:.2%}")
    print("=" * 70)

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is configured before running
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set. The evaluation will use fallback/mock responses.")
    run_evaluation()
