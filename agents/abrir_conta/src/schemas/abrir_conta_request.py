from pydantic import BaseModel


class AbrirContaRequest(BaseModel):
    message: str