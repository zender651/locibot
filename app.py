

from langchain.llms import Ollama
from fastapi import FastAPI

llm = Ollama(model="llama2")

from langchain.prompts.prompt import PromptTemplate


template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.
Please be concise.
<</SYS>>


Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %}
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)

app=FastAPI()

@app.post("/query")
def predict(message: str, history: str):
    response = conversation.predict(input=message)

    return response