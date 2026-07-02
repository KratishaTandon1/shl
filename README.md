---
title: shl
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Conversational SHL Assessment Recommender API
**Take-home Assignment for AI Intern Role — SHL Labs**

A stateless FastAPI service that provides a conversational agent to guide recruiters from a vague hiring intent to a grounded shortlist of SHL assessments through dialogue. The recommender clarifies vague queries, accepts modifications, handles comparative questions, and refuses off-topic requests—all while ensuring strict schema compliance and relying exclusively on the SHL product catalog.

---

## 🚀 Key Features
- **Stateless & Scalable Design**: Every conversational turn is stateless; the client carries the full history. 
- **Hybrid Pinning & TF-IDF Search Engine**: Bypasses semantic dilution by combining a pure-Python TF-IDF Vector Space search with intent-based substring matches (handling queries like `REST` $\rightarrow$ `RESTful Web Services`).
- **Topic & Domain Expansion**: Programmatically expands general role queries (e.g. `finance`) to encompass related technical and domain keywords (e.g. `statistics`, `accounting`, `math`), ensuring maximum target recall.
- **Defensive Dialogue Guardrails**: Enforces empty-list protection (`recommendations: []`) during intermediate turns, comparative responses, or scope refusals.
- **Zero-Item Resolution Guard**: Intercepts cases where proposed recommendations cannot be resolved to the catalog, returning a clarifying response and setting `end_of_conversation: false`.
- **FastAPI & Pydantic Validation**: Strict schema adherence on inputs and outputs to guarantee compliance with the evaluator harness.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KratishaTandon1/shl.git
cd shl
```

### 2. Set Up a Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory (or update the existing one) with your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 💻 Running the Service Locally

Start the local Uvicorn development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The service will be available at `http://localhost:8000`.

---

## 🔌 API Specification

### 1. Health Check
Ready-check endpoint for cold starts.
- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Conversational Chat
Processes the full conversation history and returns the next agent turn.
- **Endpoint**: `POST /chat`
- **Request Payload**:
  ```json
  {
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java developer."},
      {"role": "assistant", "content": "Sure, what is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years"}
    ]
  }
  ```
- **Response Payload**:
  ```json
  {
    "reply": "Got it. Here is the shortlist for a mid-level Java developer.",
    "recommendations": [
      {
        "name": "Core Java (Advanced Level) (New)",
        "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "test_type": "K"
      }
    ],
    "end_of_conversation": false
  }
  ```

---

## 🧪 Running the Verification & Test Suite

We provide a robust suite of local evaluation tools to test name resolution, retrieval recall, and dialogue compliance:

### 1. Catalog Name Resolution
Verifies that all recommended assessments cleanly map back to the official catalog database and correct test types:
```bash
python test_resolution.py
```

### 2. Retrieval Recall Test
Verifies that the hybrid retrieval engine retrieves 100% of required trace assessments in the top-100 context window:
```bash
python verify_retrieval.py
```

### 3. Full Offline Evaluation Replay
Simulates actual dialogue trees against the local FastAPI endpoint. Under the hood, this sets `LOCAL_EVAL_MODE=true` to fetch cached mock responses, bypassing the Gemini API free-tier 429 rate limit quota:
```bash
python run_eval.py
```

### 4. Regenerating Cache
If you make model prompt adjustments and want to rebuild the offline test cache:
```bash
python generate_cache_from_traces.py
```
