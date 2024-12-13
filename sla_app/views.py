from flask import render_template, request, jsonify, session, Response, Flask, stream_with_context
from sla_app.modules.rag import ChatGroq, ChatMistralAI
from sla_app.modules.agent import PydanticOutputParser
import sla_app.modules.agent.utils as agent
import sla_app.modules.rag.utils as rag
from flask_session import Session
from sla_app import app as server
from dotenv import load_dotenv
import pandas as pd
import threading
import time
import json
import re
import os # pour la clé de l'api

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

_ = load_dotenv()

data_path = "sla_app/data"

increment = 0

src_files = {
    "droit_foncier": os.path.join(data_path, "foncier_et_domanial.csv"),
    "marches_publics": os.path.join(data_path, "marches_publics.csv"),
    "procedures_penales": os.path.join(data_path, "procedures_penales.csv"),
    "penal": os.path.join(data_path, "penal.csv"),
    "collectivites_locales": os.path.join(data_path, "collectivites_locales.csv"),
    "famille": os.path.join(data_path, "famille.csv")
}

src_files_list = [
    src_files["droit_foncier"],
    src_files["marches_publics"],
    src_files["procedures_penales"],
    src_files["penal"],
    src_files["collectivites_locales"],
    src_files["famille"],
]

n_logs = 0

logs = {
    "results": [],
    "nodes": [],
    "recursion_error": False
}

# a function for selection
def add_selection(selected, select_dict):
    
    options = [select_dict["sel"]]
    options.extend(select_dict["nsel"])
    
    options.remove(selected)
    
    select_dict["sel"] = selected
    
    select_dict["nsel"] = options
    
    return select_dict

# initialize hyperparameters
query = None

chunk_sizes = {
    "droit_foncier": 2631,
    "marches_publics": 5769,
    "procedures_penales": 1405,
    "penal": 2917,
    "collectivites_locales": 2499,
    "famille": 1133,
}

embedding_ids = {
    "sel": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "nsel": ["intfloat/multilingual-e5-base", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/distiluse-base-multilingual-cased-v2"]
}

load = False

base_n = 7
bm25_n = 6
max_iter = 25 
max_retries = 20
rerankers = {
    "sel": "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR",
    "nsel": ["antoinelouis/crossencoder-camembert-large-mmarcoFR"]
}
add_hybrid_search = True
multiple_query_llm = None
base_weight = 0.8
metadata_positions = {
    "sel": "after",
    "nsel": ["before", "none"]
}
metrics = {
    "sel": "l2",
    "nsel": ["cosine"]
}
verbose = True
temperature = 0.4
chat_models = {
    "sel": "mistral-large-latest (mistral)",
    "nsel": ["llama-3.1-70b-versatile (groq)", "llama-3.1-8b-instant (groq)", "mixtral-8x7b-32768 (groq)",
             "open-mistral-nemo (mistral)", "open-mistral-7b (mistral)", "open-mixtral-8x22b (mistral)"]
}
tr_models = {
    "sel": "llama-3.1-70b-versatile (groq)",
    "nsel": ["mistral-large-latest (mistral)"]
}
response = ""

# other hyperparameters
targets = {
    "sel": "citizen",
    "nsel": ["expert"]
}

# initialize prompts
ref_template = """
Think step by step and extract all legal references from the context.
{format_instructions}
context={context}
"""

SEARCH_PROMPT = """Act as a question reformulator and perform the following task:
- Transform the following input question into an improved version, optimized for web search.
- When reformulating, examine the input question and try to reason about the underlying semantic intention/meaning.
- The output question must be written in French.
- Provide only the output question and nothing else.
- Note that the question is asked by a Senegalese person.
- Add more context to the request if necessary.
"""

QUERY_PROMPT = """Act as a question reformulator and perform the following task:
- Transform the following input question into an improved version.
- When reformulating, examine the input question and try to reason about the underlying semantic intent/meaning.
- The output question must be written in French.
- Provide only the output question and nothing else.
- Think step by step.
"""

DECISION_PROMPT = """You are a decision-making expert based on a context obtained from a query.
Follow these instructions to decide:
- If the number of reference documents (defined at the beginning of the context) is strictly greater than 0, if the number of attempts to extract references (defined at the beginning of the context) does not exceed 3, and the context before the web-extracted context contains any reference to legal articles, laws, decrees, legal textbooks, legal text titles, legal text chapters, legal text sections, legal text subsections, or paragraphs, then return 'references'.
- Otherwise, if the context contains no references and is not clear enough or if the filtering percentage exceeds 70% and the number of search attempts (defined at the beginning of the context) does not exceed 3, return 'search'.
- Otherwise, if the context contains no references but has enough information to answer the request, return 'final_answer'.
- Be especially careful not to exceed the limits on the number of attempts.
- Think step by step before providing your decision.
"""

c_prompt = """You are a question-answering assistant for Senegalese law, aimed at Senegalese citizens. Use simple language in French. Respond only using the provided context. If no relevant information is available, clearly indicate that you do not know the answer. Do not create unfounded or speculative responses.

1. Use of context: Respond exclusively based on the elements provided in the context. Do not supplement with external information or assumptions.
2. Precise references: Include only exact references (laws or decrees) without mentioning chapters or titles. If a reference is not provided, do not invent one.
3. Verification: Ensure that all sentences are correct and that references are precise. Avoid ambiguities.
4. Clarity and precision: Provide detailed, correct, coherent, and concise answers. Leave no room for interpretation.
5. The additional articles of an article refer to those to which they relate.
6. Think step by step before providing your final answer.
""" 

e_prompt = """You are a question-answering assistant for experts in Senegalese law. Use language suitable for professionals and provide all answers in French. Respond only using the provided context.

1. Use of context: Respond exclusively based on the provided elements. Do not supplement with external information or assumptions.
2. Precise references: Include only exact references (laws or decrees) without mentioning chapters or titles. If a reference is not present, do not invent one.
3. Clarity and precision: Ensure that all sentences are correct and that references are precise. Avoid ambiguities.
4. Incomplete answers: If no relevant information is available in the context, clearly indicate that you do not know the answer. Do not fabricate responses.
5. Details and completeness: Provide detailed, correct, coherent, and concise answers. Prioritize accuracy and reliability.
6. The additional articles of an article refer to those to which they relate.
7. Think step by step before providing your final answer.
"""



# initialize metadata
domaine = ""
loi = ""
decret = ""
arrete = ""
declaration = ""
partie = ""
livre = ""
titre = ""
sous_titre = ""
chapitre = ""
section = ""
sous_section = ""
application = ""
loyer = ""
localite = ""
categorie = ""
habitation = ""

# recuperate data for metadata filtering
articles = pd.read_csv('sla_app/data/articles.csv')

# main columns
columns = ['domaine', 'numero_loi', 'numero_decret', 'numero_arrete',
            'declaration', 'division_partie', 'division_livre',
            'division_titre', 'division_sous_titre', 'division_chapitre',
            'division_section', 'division_sous_section', 'section_application',
            'loyer', 'localite', 'categorie',
            'type_habitation'
        ]

# column_map
column_map = {
    'numero_loi': 'loi',
    'numero_decret': 'decret',
    'numero_arrete': 'arrete',
    'division_partie': 'partie',
    'division_livre': 'livre',
    'division_titre': 'titre',
    'division_sous_titre': 'sous_titre',
    'division_chapitre': 'chapitre',
    'division_section': 'section',
    'division_sous_section': 'sous_section',
    'section_application': 'application',
    'type_habitation': 'habitation'
}

articles = articles[columns]

articles['declaration'] = articles['declaration'].map(lambda x: ['non', 'oui'][x])
articles['loyer'] = articles['loyer'].map(lambda x: ['non', 'oui'][x])

articles.fillna('', inplace = True)

# change domaine value '' to 'Autres'
articles['domaine'] = articles['domaine'].map(lambda x: 'Autres' if x == '' else x)

# parameters related to threads
result_offset = 0
log_offset = 0

@server.route("/")    
@server.route("/index")    
def index():
    
    try:
        
        return render_template("index.html", title = "Sen Legal Assistant", page = 'index')
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )
        
@server.route("/services")    
def services():
    
    try:
        
        return render_template("services.html", title = "Services", page = 'service')
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )

@server.route("/pricing")    
def pricing():
    
    try:
        
        return render_template("pricing.html", title = "Prix", page = 'price')
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )
        
@server.route("/contacts")    
def contacts():
    
    try:
        
        return render_template("contacts.html", title = "Contacts", page = 'contact')
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )
        
@server.route("/signup")    
def signup():
    
    try:
        
        return render_template("signup.html", title = "S'inscrire", page = 'inscription')
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )

def substitute_strong(text):

    # Define the regex pattern  
    pattern = r'\*\*(.*?)\*\*'  

    # Substitute the matched pattern with the bold HTML tags  
    output_string = re.sub(pattern, r'<strong>\1</strong>', text)
    
    return output_string  


def rag_generator():
    
    global base_n
    global bm25_n
    global rerankers
    global add_hybrid_search
    global multiple_query_llm
    global base_weight
    global metadata_positions
    global metrics
    global load
    
    print("Entering rag context generation")
    
    # instantiate filter model
    filter_llm = ChatGroq(model="mixtral-8x7b-32768")
    
    chunk_sizes_list = [
        chunk_sizes["droit_foncier"],
        chunk_sizes["marches_publics"],
        chunk_sizes["procedures_penales"],
        chunk_sizes["penal"],
        chunk_sizes["collectivites_locales"],
        chunk_sizes["famille"],
    ]
    
    # get embeddings
    embeddings = rag.get_embedding(
        embedding_ids["sel"]
    )
    
    base_n = int(base_n)
    bm25_n = int(bm25_n)
    base_weight = float(base_weight)
    reranker = rerankers["sel"]
    if reranker != "none":
        rerank = False
    top_rerank = base_n + bm25_n
    add_hybrid_search = add_hybrid_search
    multiple_query_llm = multiple_query_llm
    include_original = True
    base_weight = base_weight
    bm25_weight = 1 - base_weight
    metadata_position = metadata_positions["sel"]
    if metadata_position == "after":
        add_metadata_before = False
        add_metadata = True
    elif metadata_position == "before":
        add_metadata_before = True
        add_metadata = True
    else:
        add_metadata = False
        add_metadata_before = False

    
    documents = rag.get_documents_from_files(
        src_files=src_files_list,
        chunk_sizes=chunk_sizes_list,
        add_metadata=add_metadata,
        add_metadata_before=add_metadata_before,
        add_contextual_container=True,
    )
    
    db = rag.get_faiss_vectorstore(documents, embeddings, metric = metrics["sel"], load=load)
    
    base_retriever = rag.get_document_extractor(db, n=base_n)  # semantic_similarity retriever
    
    retriever, filters = rag.get_base_retriever_and_filters(
        filter_llm,
        base_retriever,
        documents,
        embeddings,
        multiple_query_llm=multiple_query_llm,
        n=bm25_n,
        rerank=rerank,
        reranker=reranker,
        top_rerank=top_rerank,
        add_hybrid_search=add_hybrid_search,
        include_original=include_original,
        base_weight=base_weight,
        bm25_weight=bm25_weight,
    )  # retriever with advanced filtering
    
    return retriever, filters, documents, db, embeddings
    
def chat_generator():
    
    # initialize some hyperparameters
    global logs
    global query
    global chunk_sizes
    global embedding_ids
    global base_n
    global bm25_n
    global max_iter
    global rerankers
    global add_hybrid_search
    global multiple_query_llm
    global base_weight
    global metadata_positions
    global metrics
    global verbose
    global temperature
    global chat_models
    global tr_models
    global response
    global ref_template
    global targets
    global c_prompt
    global e_prompt
    global QUERY_PROMPT
    global SEARCH_PROMPT
    global DECISION_PROMPT
    global max_retries
    
    print("Entering chat thread")
    
    llm_query_rewriter_ = agent.get_query_rewriter(
        ChatMistralAI(model="open-mistral-nemo"), QUERY_PROMPT
    )
    
    retriever, filters, documents, db, embeddings = rag_generator()
    
    llm_mistral_dec = agent.get_mistral_llm_for_decision("mistral-large-latest")
    
    mistral_decider = agent.get_decider(llm_mistral_dec, sys_prompt=DECISION_PROMPT)
    
    parser = PydanticOutputParser(pydantic_object=agent.References)
    
    ref_llm = ChatMistralAI(model="mistral-large-latest")

    ref_retriever = agent.get_reference_retriever(ref_llm, parser, template=ref_template)

    triples_llm = ChatGroq(model=tr_models["sel"].replace("(groq)", "").strip(), temperature=temperature) if "groq" in tr_models["sel"]\
        else ChatMistralAI(model=tr_models["sel"].replace("(mistral)", "").strip(), temperature=temperature)
    
    prompt_template = agent.get_triples_prompt_from_file(
        "sla_app/data/triples_prompt_short.txt"
    )
    
    llm_query_rewriter = agent.get_web_search_query(
        ChatMistralAI(model="mistral-large-latest"), SEARCH_PROMPT
    )
    
    chat_model = ChatGroq(model=chat_models["sel"].replace("(groq)", "").strip(), temperature=temperature) if "groq" in chat_models["sel"]\
        else ChatMistralAI(model=chat_models["sel"].replace("(mistral)", "").strip(), temperature=temperature)
    
    agent_system = agent.AgentSystemv1(
        documents,
        embeddings,
        db,
        retriever,
        filters,
        mistral_decider,
        ref_retriever,
        triples_llm,
        prompt_template,
        llm_query_rewriter_,
        llm_query_rewriter,
        verbose=verbose,
        chat_llm=chat_model,
        logs=logs,
        c_prompt=c_prompt,
        e_prompt=e_prompt,
        target=targets["sel"],
        max_retries=max_retries
    )
    
    # sending the query to the multi-agent system
    try:
        result = agent_system.invoke({"query": query}, config = {'recursion_limit': max_iter})
        response = result['answer']
    except RecursionError:
        logs["results"].append([{"log": "Maximum number of iterations reached ! Try to augment the maximum number of iterations.", "color": "red"}])
        logs["nodes"].append("Error Criterion :")
    except rag.RateLimitError as rl:
        if rl.task == "answer":
            model = chat_models["sel"]
            task = "chat"
        elif rl.task == "triples":
            model = tr_models["sel"]
            task = "extraction of triplets"
        elif rl.task == "query_rewrite_search":
            model = "Non Specified"
            task = "rewrite query or question for web search"
        elif rl.task == "query_rewrite":
            model = "Non Specified"
            task = "rewrite query or question"
        elif rl.task == "references":
            model = "Non Specified"
            task = "extraction of references"
        elif rl.task == "search":
            model = "Non Specified"
            task = "web search"
        logs["results"].append([{"log": f"An API error is occured ! Choose another model for the task '{task}' different from the model {model}.", "color": "red"}])
        logs["nodes"].append("Error Criterion :")
    
def log_generator():
    
    # initialize some hyperparameters
    global query
    global chunk_sizes
    global embedding_ids
    global base_n
    global bm25_n
    global max_iter
    global rerankers
    global add_hybrid_search
    global multiple_query_llm
    global base_weight
    global metadata_positions
    global metrics
    global targets
    global verbose
    global temperature
    global chat_models
    global tr_models
    global response
    global ref_template
    global logs
    global n_logs
    global result_offset
    global log_offset
    global c_prompt
    global e_prompt
    global QUERY_PROMPT
    global SEARCH_PROMPT
    global DECISION_PROMPT
    global max_retries
    
    logs["results"] = []
    
    logs["nodes"] = []
    
    result_offset = 0
    log_offset = 0
    
    print("Entering log thread")
    
    while True:
        
        if len(logs["results"]) > result_offset:
            
            n_logs = len(logs["nodes"])
            
            result = logs["results"][result_offset]
            
            node = logs["nodes"][result_offset]
            
            print(len(result), log_offset, result_offset)
            if len(result) >= log_offset:
                
                if len(result) > 0:
                
                    if node == "":
                        
                        if len(result) > 0 : query = result[0]["log"].replace("New Query =>", "").strip()
                    
                    if node == "Final Answer :":
                        
                        log = result[log_offset]
                        
                        log["log"] = substitute_strong(log["log"].replace("Answer =>", "").strip())
                        
                        dict_ = {
                            "result": log,
                            "node": node,
                            "query": query,
                            "temperature": temperature,
                            "base_n": base_n,
                            "bm25_n": bm25_n,
                            "base_weight": base_weight,
                            "max_iter": max_iter,
                            "target": targets["sel"],
                            "chat_model": chat_models["sel"],
                            "tr_model": tr_models["sel"],
                            "embedding_id": embedding_ids["sel"],
                            "metric": metrics["sel"],
                            "reranker": rerankers["sel"],
                            "c_prompt": c_prompt, 
                            "e_prompt": e_prompt,
                            "q_prompt": QUERY_PROMPT,
                            "s_prompt": SEARCH_PROMPT,
                            "d_prompt": DECISION_PROMPT,
                            "max_retries": max_retries
                        }
                        
                        yield f"data: {json.dumps(dict_)}\n\n"
                        
                        break
                    
                    elif node == "Error Criterion :":
                    
                        log = result[log_offset]
                        
                        dict_ = {
                            "result": log,
                            "node": node,
                            "query": query,
                            "temperature": temperature,
                            "base_n": base_n,
                            "bm25_n": bm25_n,
                            "max_iter": max_iter,
                            "base_weight": base_weight,
                            "target": targets["sel"],
                            "chat_model": chat_models["sel"],
                            "tr_model": tr_models["sel"],
                            "embedding_id": embedding_ids["sel"],
                            "metric": metrics["sel"],
                            "reranker": rerankers["sel"],
                            "c_prompt": c_prompt, 
                            "e_prompt": e_prompt,
                            "q_prompt": QUERY_PROMPT,
                            "s_prompt": SEARCH_PROMPT,
                            "d_prompt": DECISION_PROMPT,
                            "max_retries": max_retries
                        }
                        
                        yield f"data: {json.dumps(dict_)}\n\n"
                        
                        break
            
                if log_offset + 1 < len(result):
                    
                    dict_ = {
                        "result": result[log_offset],
                        "node": node
                    }
                
                    yield f"data: {json.dumps(dict_)}\n\n"
                    
                    log_offset += 1
                    
                elif len(logs["results"]) > result_offset + 1:
                    
                    if len(result) > 0:
                        
                        dict_ = {
                            "result": result[log_offset],
                            "node": node
                        }
                    
                        yield f"data: {json.dumps(dict_)}\n\n"
                    
                    result_offset += 1
                    
                    log_offset = 0
                    
        time.sleep(2)
        
@server.route("/stream-agent")
def stream_agent():
    
    threading.Thread(target=log_generator, daemon=True).start()
    threading.Thread(target=chat_generator, daemon=True).start()
    
    return Response(stream_with_context(log_generator()), mimetype='text/event-stream')

@server.route("/rag_system/", methods = ["POST"])
@server.route("/rag_system/")
def rag_system():
    
    # initialize some hyperparameters
    global logs
    global query
    global chunk_sizes
    global embedding_ids
    global base_n
    global bm25_n
    global rerankers
    global add_hybrid_search
    global multiple_query_llm
    global base_weight
    global metadata_positions
    global metrics
    global verbose
    global temperature
    global chat_models
    global response
    global targets
    global ref_template
    global articles
    global c_prompt
    global e_prompt
    global QUERY_PROMPT
    global load
    global max_retries
    
    # initialize the context
    contexts = []
    
    try:
        
        if request.method == "POST":

            query = request.form.get('query')
            
            base_n = request.form.get('base_n')
            
            base_n = int(base_n)
            
            bm25_n = request.form.get('bm25_n')
            
            bm25_n = int(bm25_n)
            
            max_retries = request.form.get('max_retries')
            
            max_retries = int(max_retries)
            
            temperature = request.form.get('temperature')
            
            temperature = float(temperature)
            
            base_weight = request.form.get('base_weight')
            
            base_weight = float(base_weight)
            
            target = request.form.get('target')
            
            targets = add_selection(target, targets)
            
            chat_model = request.form.get('chat_model')
            
            chat_models = add_selection(chat_model, chat_models)
            
            embedding_id = request.form.get('embedding_id')
            
            embedding_ids = add_selection(embedding_id, embedding_ids)
            
            metric = request.form.get('metric')
            
            metrics = add_selection(metric, metrics)
            
            if not session.get("metric") and not session.get("embedding_id"):
                
                session["metric"] = metrics["sel"]
                
                session["embedding_id"] = embedding_ids["sel"]
                
            elif metrics["sel"] == session.get("metric") and embedding_ids["sel"] == session.get("embedding_id"):
                
                load = True
            
            else:
                
                load = False
            
            reranker = request.form.get('reranker')
            
            rerankers = add_selection(reranker, rerankers)
            
            c_prompt = request.form.get('c_prompt')
            
            e_prompt = request.form.get('e_prompt')
            
            QUERY_PROMPT = request.form.get('q_prompt')
                    
            if query != '' and not query is None:
                
                llm_query_rewriter_ = agent.get_query_rewriter(
                    ChatMistralAI(model="open-mistral-7b"), QUERY_PROMPT
                )
                
                try:
                    
                    @rag.execute_with_count(max_retries, "query_rewrite")
                    def execute(query):

                        query = llm_query_rewriter_.invoke(query)
                        
                        return query
                    
                    query = execute(query)
                    
                    retriever, filters, documents, db, embeddings = rag_generator()

                    documents = retriever.invoke(query)

                    contexts = [doc.page_content for doc in retriever.invoke(query)]
                    
                    chat_llm = ChatGroq(model=chat_models["sel"].replace("(groq)", "").strip(), temperature=temperature) if "groq" in chat_models["sel"]\
                        else ChatMistralAI(model=chat_models["sel"].replace("(mistral)", "").strip(), temperature=temperature)
                    
                    # sending the query to the RAG system
                    response = rag.get_answer(
                        chat_llm,
                        query,
                        documents,
                        targets["sel"],
                        c_prompt,
                        e_prompt,
                        max_retries
                    )
                    
                except rag.RateLimitError as rl:
                    if rl.task == "query_rewrite":
                        model = "Non Specified"
                        task = "rewrite query or question"
                    elif rl.task == "answer":
                        model = chat_models["sel"]
                        task = "chat"
                    response = f"<i class='text-danger'>An API error occured ! Choose another model for the task '{task}' different from the model {model}.</i>"
                
            else:
                
                response = "You must write a query to get an answer from the system !"
            
            return jsonify({"title": "Answer from the RAG system", "response":substitute_strong(response.replace('\n', '<br>')),
                            "context": [text.replace('\n', '<br>') for text in contexts], "query": query, 
                            "correct": len(contexts) > 0, "result": True, "temperature": temperature, "base_weight": base_weight, "base_n": base_n, 
                            "bm25_n": bm25_n, "chat_model": chat_models["sel"], 
                            "embedding_id": embedding_ids["sel"], "metric": metrics["sel"],
                            "reranker": rerankers["sel"], "target": targets["sel"],
                            "c_prompt": c_prompt, "e_prompt": e_prompt, "q_prompt": QUERY_PROMPT, "max_retries": max_retries})
        
        else:
            
            return render_template("rag_system.html", result = False, title = "RAG system", page = 'rag', base_n = base_n, bm25_n = bm25_n, chat_models = chat_models,
                                   embedding_ids = embedding_ids, metrics = metrics, rerankers = rerankers, targets = targets,
                                   temperature = temperature, base_weight = base_weight, c_prompt = c_prompt, e_prompt = e_prompt,
                                   q_prompt = QUERY_PROMPT, max_retries = max_retries
                                   )
            
    except Exception as e:
        agent.col_print(e, agent.TextColors.RED)
        return render_template(
            "error.html",
            title="Error",
            error=e,
            result=False,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )
        
@server.route("/agent_system/", methods = ["POST"])
@server.route("/agent_system/")
def agent_system():
    
    # initialize some hyperparameters
    global articles
    global query
    global logs
    global chunk_sizes
    global embedding_ids
    global base_n
    global bm25_n
    global max_iter
    global rerankers
    global add_hybrid_search
    global multiple_query_llm
    global base_weight
    global metadata_positions
    global metrics
    global verbose
    global targets
    global temperature
    global chat_models
    global tr_models
    global response
    global c_prompt
    global e_prompt
    global QUERY_PROMPT
    global SEARCH_PROMPT
    global DECISION_PROMPT
    global load
    global max_retries
     
    try:
        
        if request.method == "POST":

            query = request.form.get('query')
            
            base_n = request.form.get('base_n')
            
            base_n = int(base_n)
            
            bm25_n = request.form.get('bm25_n')
            
            bm25_n = int(bm25_n)
            
            max_retries = request.form.get('max_retries')
            
            max_retries = int(max_retries)
            
            max_iter = request.form.get('max_iter')
            
            max_iter = int(max_iter)
            
            temperature = request.form.get('temperature')
            
            temperature = float(temperature)
            
            base_weight = request.form.get('base_weight')
            
            base_weight = float(base_weight)
            
            target = request.form.get('target')
            
            targets = add_selection(target, targets)
            
            chat_model = request.form.get('chat_model')
            
            chat_models = add_selection(chat_model, chat_models)
            
            tr_model = request.form.get('tr_model')
            
            tr_models = add_selection(tr_model, tr_models)
            
            embedding_id = request.form.get('embedding_id')
            
            embedding_ids = add_selection(embedding_id, embedding_ids)
            
            metric = request.form.get('metric')
            
            metrics = add_selection(metric, metrics)
            
            if not session.get("metric") and not session.get("embedding_id"):
                
                session["metric"] = metrics["sel"]
                
                session["embedding_id"] = embedding_ids["sel"]
                
            elif metrics["sel"] == session.get("metric") and embedding_ids["sel"] == session.get("embedding_id"):
                
                load = True
            
            else:
                
                load = False
            
            reranker = request.form.get('reranker')
            
            rerankers = add_selection(reranker, rerankers)
            
            c_prompt = request.form.get('c_prompt')
            
            e_prompt = request.form.get('e_prompt')
            
            QUERY_PROMPT = request.form.get('q_prompt')
            
            SEARCH_PROMPT = request.form.get('s_prompt')
            
            DECISION_PROMPT = request.form.get('d_prompt')
                    
            if query != '' and not query is None:
                
                correct = True
                
            else:
                
                response = "You must write a query to get an answer from the system !"

                correct = False
            
            return jsonify(
                    {
                        'correct': correct,
                        'result': True,
                        'title': "Answer from the Multi-Agent system"
                    }
            )
        
        else:
            
            return render_template("agent_system.html", result = False, title = "Multi-Agent System", page = 'agent', base_n = base_n, bm25_n = bm25_n, max_iter = max_iter, 
                                   temperature = temperature, base_weight = base_weight, chat_models = chat_models, tr_models = tr_models, targets = targets,
                                   embedding_ids = embedding_ids, metrics = metrics, rerankers = rerankers,
                                   c_prompt = c_prompt, e_prompt = e_prompt, q_prompt = QUERY_PROMPT,
                                   s_prompt = SEARCH_PROMPT, d_prompt = DECISION_PROMPT, max_retries = max_retries
                                   )
            
    except Exception as e:
        return render_template(
            "error.html",
            title="Error",
            error=e,
            result=False,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )



