from fastapi import FastAPI

from fastapi.exceptions import HTTPException

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain.chat_models import init_chat_model


from dotenv import load_dotenv


from src.schemas import AbrirContaRequest

import os

import logging


load_dotenv()

_llm = init_chat_model(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()



agente_abertura_conta = create_agent(
    _llm,
    tools=[],
    system_prompt=(
         "Você é um especialista em abertura de contas do banco MDBank. "
        "Ajude o cliente a abrir uma conta e explique os tipos disponíveis."
    )
)




@app.post("/send")
async def consultar(request: AbrirContaRequest):
    mensagem = request.message

    if not mensagem:
        raise HTTPException(
            status_code=400,
            detail="Campo 'message' obrigatório"
        )


    try:
        resultado = agente_abertura_conta.invoke(
            {"messages": [HumanMessage(content=mensagem)]}
        )

        mensagem_ia = resultado["messages"][-1]
        return {"resposta": mensagem_ia.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

