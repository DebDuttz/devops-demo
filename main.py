from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os

app = FastAPI(title="ReAct Agent API", version="1.0.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

def get_agent():
    # your Day-7 agent, built once and reused
    return ChatGroq(model="llama-3.3-70b-versatile",
                    api_key=os.environ["GROQ_API_KEY"], temperature=0)

@app.post("/ask", response_model=AskResponse)      # 📍 typed in AND out
async def ask(req: AskRequest, agent=Depends(get_agent)):
    result = agent.invoke(req.question)             # (your ReAct loop here)
    return AskResponse(answer=result.content)
