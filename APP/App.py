import os 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import TavilySearchResults
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ['LangChain_API_Key']=os.getenv('LangChain_API_Key')
groq_api_key =os.getenv('Groq_API_Key')
os.environ['TAVILY_API_KEY'] =os.getenv('TAVILY_API_KEY')



# List of PDF file paths
urls = [
    r"E:\Generative AI_LLMS\LangChain\RAG\RAG_From_Scratch\Adaptive_RAG\Data\istanbul-city-guide.pdf",
    r"E:\Generative AI_LLMS\LangChain\RAG\RAG_From_Scratch\Adaptive_RAG\Data\istanbul-city-guide.pdf"
]

# Load PDFs and flatten the list
documents = []
for url in urls:
    documents.extend(PyPDFLoader(url).load())  # Ensure we get a flat list of Document objects

# Text splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
docs = splitter.split_documents(documents)  # No more AttributeError!

# Chroma vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="./chroma_db"
)


# # Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# router

llm = ChatGroq(model ='mistral-saba-24b',groq_api_key =groq_api_key)

class RoutQuery(BaseModel):
    datasource:Literal['vectorstore','web_search']=Field(
        description ='Given a user question choose to route it to vectorstore or web_search'
    )

structured_llm_router =llm.with_structured_output(RoutQuery)

system = '''You are an expert in routing queries to the appropriate source.
The vectorstore contains documents related to the **Istanbul Guide**, covering topics such as history, neighborhoods, landmarks, tourism, entertainment, transportation, shopping, and dining.
Use the vectorstore for questions related to the **Istanbul Guide**; otherwise, use web_search.'''



rout_prompt =ChatPromptTemplate.from_messages(
    [
        ('system',system),
        ('human','{question}')
    ]
)

question_router_chain = rout_prompt | structured_llm_router

# print(question_router_chain.invoke({'question':"What percentage of customers are female vs. male?"}))
# print(question_router_chain.invoke({'question':" "}))

## Retrival Grader
class GradDocument(BaseModel):
    binary_score:str =Field(
        description ="Document are relevant to question ,'yes' or 'No'"
    )

structured_llm_grader =llm.with_structured_output(GradDocument)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

prompt_grader =ChatPromptTemplate.from_messages([
    ('system',system),
    ('human','retrived document {document} \n\n user question:{question}')
])

retrival_grader  = prompt_grader | structured_llm_grader

question ="what is the available products?"
docs =retriever.invoke(question)
doc_text =docs[1].page_content
# print(retrival_grader.invoke({'document':doc_text,'question':question}))

# Generation

generate_prompt = hub.pull('rlm/rag-prompt')

rag_chain = (
    generate_prompt
    | llm
    | StrOutputParser()
)
generation = rag_chain.invoke({'context': docs, 'question': question})
# print(generation)

# Grad Hallucination
class GradHallucination(BaseModel):
    binary_score:str =Field(
        description ="Answer is grounded in facts ,'yes' or 'no'"
    )

structured_llm_grader_H =llm.with_structured_output(GradHallucination)

# prompt 
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

prompt =ChatPromptTemplate.from_messages(
    [
        ('system',system),
        ('human','set of facts {documents} \n\n llm generation {generation}')
    ]
)

hallucination_grader_chain =prompt | structured_llm_grader_H 

# grad Answer
class GradAnswer(BaseModel):
    binary_score:str =Field(
        description ="Answer address question,'yes' or 'no'"
    )

structured_llm_grader_Ans =llm.with_structured_output(GradAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

prompt_answer_grader =ChatPromptTemplate.from_messages(
    [
        ('system',system),
        ('human','generation {generation} and question {question}')
    ]
)

answer_grader_chain =prompt_answer_grader| structured_llm_grader_Ans

# Question rewrite
# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
     Return only the improved question without any additional text or explanation."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewirter =re_write_prompt | llm | StrOutputParser()

# web school tool 
web_search_tool =TavilySearchResults(max_results = 5)

# construct graph 

from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question:str
    documents:List[str]
    generation:str

# Graph flow 

from langchain.schema import Document 

def retriever_fun(state):
    print('___Retriever___')
    question = state['question']
    documents = retriever.invoke(question)

    return {'documents': documents, 'question': question}

def generate(state):
    print('___Generate___')

    question = state['question']
    documents = state['documents']
    generation = rag_chain.invoke({'context': documents, 'question': question})

    return {'documents': documents, 'question': question, 'generation': generation}

def grade_documents(state):
    print('___Check if document is relevant to question or not____')
    
    question = state['question']
    documents = state['documents']
    
    filtered_docs = []

    for doc in documents:
        score = retrival_grader.invoke({'document': doc, 'question': question})
        grade = score.binary_score

        if grade == 'yes':
            print('___Document is relevant')
            filtered_docs.append(doc)
        else:
            print('__Document is not relevant')

    return {'documents': filtered_docs, 'question': question}

def transform_query(state):
    print('___Rewrite Query___')
    question = state['question']
    documents = state['documents']
    
    better_question = question_rewirter.invoke({'question': question})
    
    return {'documents': documents, 'question': better_question}

def web_search(state):
    print('__Web Search__')
    question = state['question']

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    print(web_results)

    return {"documents": [web_results], "question": question}

# Edges 
def RouteQuestion(state):
    '''Route question for web search or RAG'''
    print('__Route Question__')
    
    question = state['question']
    source = question_router_chain.invoke({'question': question})

    if source.datasource == 'web_search':
        print('Route question to web search')
        return 'websearch'

    elif source.datasource == 'vectorstore':
        print('Route question to vectorstore')
        return 'vectorstore'      

def Decide_to_generate(state):
    print('__Decide to generate or rewrite query__')

    question = state['question']
    filtered_docs = state['documents']

    if not filtered_docs:
        print('__All documents are not relevant, transforming query__')
        return 'transform_query'
    else:
        print('Decision: Generate')
        return 'generate'

def grade_generation_v_document_question(state):
    '''Determine if the generation is grounded in the document and if it answers the question'''
    
    print('__Check Hallucination__')

    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader_chain.invoke({'documents': documents, 'generation': generation})  
    grade = score.binary_score

    if grade == 'no':
        print('__Generation is grounded in document__')

        # Check if the generation answers the question
        score = answer_grader_chain.invoke({'question': question, 'generation': generation})
        grade = score.binary_score

        if grade == 'yes':
            print('Generation addresses the question')
            return 'usefull'

        else:
            print('Generation does not address the question')
            return 'not usefull'

    else:
        print('Generation is not grounded in the document')
        return 'not supported'

# Compile Graph

from langgraph.graph import START,END,StateGraph

workflow =StateGraph(GraphState)

# Define nodes 
workflow.add_node('web_search',web_search)
workflow.add_node('retriever_fun',retriever_fun)
workflow.add_node('generate',generate)
workflow.add_node('transform_query',transform_query)
workflow.add_node('grade_documents',grade_documents)

# build Graph
workflow.add_conditional_edges(
    START,
    RouteQuestion,
    {
        'websearch':'web_search',
        'vectorstore': 'retriever_fun'
    })

workflow.add_edge('web_search','generate')
workflow.add_edge('retriever_fun','grade_documents')
workflow.add_conditional_edges('grade_documents',
   Decide_to_generate,
   {
     'transform_query':'transform_query',
     'generate':'generate'
   } )

workflow.add_edge('transform_query','retriever_fun')
workflow.add_conditional_edges('generate',
grade_generation_v_document_question,
{
        "not supported": "generate",
        "usefull": END,
        "not usefull": "transform_query",
}
)

app_workflow =workflow.compile()


app =FastAPI()

class QueryRequest(BaseModel):
    question:str

class QueryResponse(BaseModel):
    answer:str

@app.post('/query',response_model =QueryResponse)
async def run_flow(request:QueryRequest):
    inputs ={'question':request.question}
    response =None
    for output in app_workflow.stream(inputs):
       for key, value in output.items():
          if isinstance(value, dict) and 'generation' in value:
                response = value['generation']
    if response is None:
        return {"answer": "No response was generated. Please check your query."}
    
    return {"answer": response}

# Make sure your FastAPI app is properly configured for deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)





