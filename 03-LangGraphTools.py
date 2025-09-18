from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# 1- Carrega API Key
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
llm_model = ChatGroq(model_name='llama-3.3-70b-versatile',temperature=0)
# 3 - Define o prompt do sistema
system_message = SystemMessage(content = """
Você é um assistente. Se o usuário pedir contas, use
a ferramenta 'somar'. Caso contrário, apenas responda normalmente
"""
)

# 4 - Definindo a ferramenta de soma
@tool("somar")
def somar(valores: str) -> str:
    """Soma dois números separados por vírgula"""
    try:
        a, b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"

# 5 - Criação do Agente com LangGraph
tools = [somar]
graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)
export_graph = graph

# 6 - Extrair a resposta final
def extrair_resposta_final(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhuma resposta final encontrada"
    
# 7 - Testando o Agente
if __name__ == "__main__":
    entrada1 = HumanMessage(content="Quanto é 8 + 5?")
    result1 = export_graph.invoke({"messages":[entrada1]})
    for m in result1["messages"]:
        print(m)
    resposta_texto_1 = extrair_resposta_final(result1)
    print("Resposta 1: ", resposta_texto_1)
    print()
    
    entrada2 = HumanMessage(content="Quem pintou a Monalisa?")
    result2 = export_graph.invoke({"messages":[entrada2]})
    for m in result2["messages"]:
        print(m)
    resposta_texto_2 = extrair_resposta_final(result2)
    print("Resposta 2: ", resposta_texto_2)