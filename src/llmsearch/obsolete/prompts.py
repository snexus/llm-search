DOLLY_PROMPT_TEMPLATE = """### Instruction:
Use the following pieces of context to answer the question at the end. If answer isn't in the context, say that you don't know, don't try to make up an answer.

### Context: 
---------------
{context}
---------------

### Question: {question}
"""


WIZARDLM10_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives an answer to the user's questions, only if it exists in the provided context, and refuses to answer otherwise.

CONTEXT:
---------------
{context}
---------------

USER: {question}
ASSISTANT:
"""

LLAMA_TEMPLATE = """### Instruction:
Use the following pieces of context to answer the question at the end. If answer isn't in the context, say that you don't know, don't try to make up an answer.

### Context: 
---------------
{context}
---------------

### Question: {question}
### Response:"""


NOUS_HERMES_TEMPLATE = """### Instruction:
Use the context provided below to answer user request at the end, using only information in the context. If answer isn't in the context, say that you don't know, don't try to make up an answer.

### Context: 
---------------
{context}
---------------

### User: {question}
### Response:"""


OPENAI_PROMPT_TEMPLATE = """"Context information is provided below. Given the context information and not prior knowledge, answer the question. If answer isn't in the context, say that you don't know, don't try to make up an answer.

### Context: 
---------------------
{context}
---------------------

### Question: {question}
"""

TULU8_TEMPLATE = """### Human: Use information in the provided context to answer the following question: ```{question}```.

### Context: 
---------------
{context}
---------------

### Assistant:
"""


REDPAJAMA_TEMPLATE = """Given the context and no prior information, answer the question.

# Context:
------------
{context}
------------
# Question: {question}
# Answer:
"""
