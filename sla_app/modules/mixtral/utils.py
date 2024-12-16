from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.tools import BaseTool
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers import SentenceTransformer as ST
from pinecone import Pinecone
import langchain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.tools import BaseTool
from sentence_transformers import SentenceTransformer as ST
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain import hub
import os

# Load the embedding model from openai
def load_embedding_model(name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    
    # chargement du modèle
    model_name = name

    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    
    return embeddings

# Recuperate Pinecone index
def load_index(index = 'legal-text-1'):
    
    pc = Pinecone(
    os.environ['PINECONE_API_KEY_'],
    environment = 'gcp-starter'
    )

    # récupération de l'index
    index = pc.Index(index)
    
    return index

# Recuperate the vector store
def load_vector_store(index, embeddings, test_key = 'contenu'):
    
    vectorstore = PineconeVectorStore(index, embeddings, text_key=test_key)

    return vectorstore


# prepare the agent
def prepare_agent(index, embeddings, llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature = 0.4, k = 5, metadata = {}):
    
    llm = HuggingFaceEndpoint(repo_id=llm_model, max_new_tokens = 10000, temperature = temperature)

    chat_model = ChatHuggingFace(llm=llm, language = 'fr')

    vectorstore = load_vector_store(index, embeddings)
    
    plot_retriever = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(k = k, search_kwargs = metadata)
    )
    
    class PlotRetrieverTool(BaseTool):
        name = "Land Plot Search"
        description = "Pour quand tu souhaites obtenir des informations supplémentaires sur le droit foncier Sénégalais en utilisant la requête en Français et non en Anglais. La question sera en chaîne de caractères. Retourne une chaîne de caractères."

        def _run(self, query: str) -> str:
            return plot_retriever.invoke(query)

        async def _arun(self, query: str) -> str:
            return await plot_retriever.ainvoke(query)

    tools = [
        PlotRetrieverTool(),
    ]
    
    # setup ReAct style prompt
    agent_prompt = hub.pull("hwchase17/react-json")
    agent_prompt = agent_prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent_prompt.messages[0] = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['tool_names', 'tools'], template='The final\
    answer must be clear and contain all the necessary information.\
        The usage of the tools must be abstract to the user.\
        Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nThe way you use the tools is by specifying\
    a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the "action"\
    field are: {tool_names}\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input":\
        $INPUT\n}}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the\
        action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question in French and not in English\n\nBegin! Reminder to always use the exact\
        characters `Final Answer` when responding.'))

    # define the agent
    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | agent_prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    # instantiate AgentExecutor
    qa = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return qa

# recuperate the response from the agent
def get_response(query, agent):
    
    return agent.invoke({
        'input': query
    })['output'].replace('</s>', '')

def filter_articles(data_frame, **metadata):
    
    domaine = metadata['domaine']
    loi = metadata['loi']
    decret = metadata['decret']
    arrete = metadata['arrete']
    declaration = metadata['declaration']
    partie = metadata['partie']
    livre = metadata['livre']
    titre = metadata['titre']
    sous_titre = metadata['sous_titre']
    chapitre = metadata['chapitre']
    section = metadata['section']
    sous_section = metadata['sous_section']
    paragraphe = metadata['paragraphe']
    loyer = metadata['loyer']
    localite = metadata['localite']
    categorie = metadata['categorie']
    habitation = metadata['habitation']
    code = metadata['code']
       
    if not domaine is None and domaine != '':
        
        data_frame = data_frame[data_frame['domaine'] == domaine]
        
    if not loi is None and loi != '':
        
        data_frame = data_frame[data_frame['loi'] == loi]
        
    if not decret is None and decret != '':
        
        data_frame = data_frame[data_frame['decret'] == decret]
        
    if not arrete is None and arrete != '':
        
        data_frame = data_frame[data_frame['arrete'] == arrete]
        
    if not declaration is None and declaration != '':
        
        data_frame = data_frame[data_frame['declaration'] == declaration]
        
    if not partie is None and partie != '':
        
        data_frame = data_frame[data_frame['partie'] == partie]
        
    if not livre is None and livre != '':
        
        data_frame = data_frame[data_frame['livre'] == livre]
        
    if not titre is None and titre != '':
        
        data_frame = data_frame[data_frame['titre'] == titre]
        
    if not sous_titre is None and sous_titre != '':
        
        data_frame = data_frame[data_frame['sous_titre'] == sous_titre]
        
    if not chapitre is None and chapitre != '':
        
        data_frame = data_frame[data_frame['chapitre'] == chapitre]
        
    if not section is None and section != '':
        
        data_frame = data_frame[data_frame['section'] == section]
        
    if not sous_section is None and sous_section != '':
        
        data_frame = data_frame[data_frame['sous_section'] == sous_section]
        
    if not paragraphe is None and paragraphe != '':
        
        data_frame = data_frame[data_frame['paragraphe'] == paragraphe]
        
    if not loyer is None and loyer != '':
        
        data_frame = data_frame[data_frame['loyer'] == loyer]
        
    if not localite is None and localite != '':
        
        data_frame = data_frame[data_frame['localite'] == localite]
        
    if not categorie is None and categorie != '':
        
        data_frame = data_frame[data_frame['categorie'] == categorie]
        
    if not habitation is None and habitation != '':
        
        data_frame = data_frame[data_frame['habitation'] == habitation]
        
    if not code is None and code != '':
        
        data_frame = data_frame[data_frame['code'] == code]
        
    return data_frame
    

def get_metadata_as_dict(data_frame):
    
    # convert to a dictionary
    data_frame = data_frame.to_dict('list')

    # recuperate uniques values
    uniques = {key: list(set(value)) for key, value in data_frame.items()}
    
    uniques = {key: [''] + [v for v in value if v != ''] for key, value in uniques.items()}
    
    # sort the values
    for key in uniques:
        
        uniques[key] = sorted(uniques[key])
    
    return uniques