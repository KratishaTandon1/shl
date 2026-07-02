import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import catalog
import agent

# Load dotenv to configure environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="SHL Assessment Recommender API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class RecommendationOutput(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationOutput]
    end_of_conversation: bool

@app.get("/health")
def health_check():
    """
    GET /health readiness check.
    Returns HTTP 200 with status: ok.
    """
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    POST /chat stateless endpoint.
    Carries the full conversation history.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages history cannot be empty")
        
    # Convert request messages to format expected by the Gemini agent
    messages_list = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Process history using agent
    agent_res = agent.chat_turn(messages_list)
    
    # Post-process recommendations to resolve official names, URLs and test types
    resolved_recs = []
    seen_urls = set()
    
    for rec in agent_res.recommendations:
        resolved = catalog.resolve_recommendation(rec.name, rec.url)
        if resolved:
            name, url, test_type = resolved
            # Guard: ensure we return unique catalog products
            if url not in seen_urls:
                seen_urls.add(url)
                resolved_recs.append(RecommendationOutput(
                    name=name,
                    url=url,
                    test_type=test_type
                ))
    
    # Safety Check: Limit recommendations size between 1 and 10 items
    # If the list is empty, we must return empty recommendations (e.g. during refusal or clarifying)
    if len(resolved_recs) > 10:
        resolved_recs = resolved_recs[:10]
        
    # Safety Check: Enforce the 8-turn conversation cap
    # The cap is 8 user + assistant turns total (8 roundtrips).
    # If user messages length in history is >= 8, force end_of_conversation to True.
    user_turns = len([m for m in request.messages if m.role == "user"])
    
    end_of_conv = agent_res.end_of_conversation
    if user_turns >= 8:
        end_of_conv = True
        
    return ChatResponse(
        reply=agent_res.reply,
        recommendations=resolved_recs,
        end_of_conversation=end_of_conv
    )
