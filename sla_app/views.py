from flask import render_template, request, jsonify, url_for, session, Response, Flask, stream_with_context
from sla_app.modules.rag import ChatGroq, ChatMistralAI
from sla_app.modules.agent import PydanticOutputParser
from langchain.prompts import PromptTemplate
import sla_app.modules.agent.utils as agent
import sla_app.modules.mixtral.utils as mix
from werkzeug.utils import secure_filename
import sla_app.modules.rag.utils as rag
from langchain_openai import ChatOpenAI
from flask_session import Session
from sla_app import app as server
from dotenv import load_dotenv
import concurrent.futures
from PIL import Image
import pandas as pd
import threading
import datetime
import time
import json
import csv
import ast
import re
import os # pour la clé de l'api
# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
# from wtforms.validators import DataRequired

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
rerankers = {
    "sel": "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR",
    "nsel": ["antoinelouis/crossencoder-camembert-large-mmarcoFR"]
}
add_hybrid_search = True
multiple_query_llm = None
base_weight = 0.8127024224401845
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
    "sel": "llama-3.1-70b-versatile (groq)",
    "nsel": ["mistral-large-latest (mistral)", "llama-3.1-8b-instant (groq)", "mixtral-8x7b-32768 (groq)",
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
    "nsel": ["citizen", "expert"]
}

# initialize prompts
ref_template = """
Think step by step and extract all legal references from the context.
{format_instructions}
context={context}
"""

SEARCH_PROMPT = """Agissez en tant que reformulateur de questions et effectuez la tâche suivante :
- Transformez la question d'entrée suivante en une version améliorée, optimisée pour la recherche sur le web.
- Lors de la reformulation, examinez la question d'entrée et essayez de raisonner sur l'intention sémantique sous-jacente / la signification.
- La question de sortie doit être rédigée en français.
- Fournissez uniquement la question de sortie et rien d'autre.
- Notez que la question est posée par une personne sénégalaise.
- Ajoutez plus de contexte à la requête si nécessaire.
"""
              
DECISION_PROMPT = """Vous êtes un expert en prise décision en fonction d'un contexte obtenu à partir d'une requête.
Suivez ces instructions pour décider :
- Si le nombre de documents de références (défini en fin de contexte) est strictement supérieur à 0, si le nombre de tentatives d'extraction de références (défini en fin de contexte) ne dépasse pas 3, et le contexte avant le contexte extrait du web contient une quelconque référence à des articles juridiques, lois, décrets, arrêté, Livres de texte juridique, Titres de texte juridique, Chapitres de texte juridique, Sections de texte juridique, Sous-sections de texte juridique ou Paragraphes, alors retourner 'references'.
- Sinon si le contexte ne contient aucune référence et n'est pas assez clair ou si le pourcentage de filtrage dépasse 70% et que le nombre de tentatives de recherche (défini en fin de contexte) ne dépasse pas 3, retourner 'search'.
- Sinon si le contexte ne contient aucune référence mais contient suffisamment d'informations pour répondre à la requête, retourner 'final_answer.
- Surtout prenez garde à ne pas dépasser les limites des nombres de tentatives.
- Réfléchissez étape par étape avant de fournir votre décision.
"""

c_prompt = """Vous êtes un assistant de question-réponse pour le droit sénégalais, destiné aux citoyens sénégalais. Utilisez un langage simple en français. Répondez uniquement en utilisant le contexte fourni. Si aucune information pertinente n'est disponible, indiquez clairement que vous ne connaissez pas la réponse. Ne créez pas de réponses non fondées ou spéculatives.

1. Utilisation du contexte : Répondez exclusivement à partir des éléments fournis dans le contexte. Ne complétez pas avec des informations extérieures ou des suppositions.
2. Références précises : Incluez uniquement des références exactes (lois ou décrets) sans mentionner de chapitres ou titres. Si une référence n'est pas fournie, n'en inventez pas.
3. Vérification : Assurez-vous que toutes les phrases sont correctes et que les références sont précises. Évitez les ambiguïtés.
4. Clarté et précision : Fournissez des réponses détaillées, correctes, cohérentes et concises. Ne laissez aucune place à l'interprétation.
6. Les articles supplémentaires d'un article sont sur ceux les quels il se refèrent.
5. Réflechissez étape par étape avant de fournir votre réponse finale.
""" 
    
e_prompt = """Vous êtes un assistant de question-réponse pour des experts en droit sénégalais. Utilisez un langage adapté aux professionnels et fournissez toutes les réponses en français. Répondez uniquement en utilisant le contexte fourni.

1. Utilisation du contexte : Répondez exclusivement à partir des éléments fournis. Ne complétez pas avec des informations extérieures ou des suppositions.
2. Références précises : Incluez uniquement des références exactes (lois ou décrets) sans mentionner de chapitres ou titres. Si une référence n'est pas présente, ne l'inventez pas.
3. Clarté et précision : Assurez-vous que toutes les phrases sont correctes et que les références sont précises. Évitez les ambiguïtés.
4. Réponses incomplètes : Si aucune information pertinente n'est disponible dans le contexte, indiquez clairement que vous ne connaissez pas la réponse. Ne fabriquez pas de réponses.
5. Détails et exhaustivité : Fournissez des réponses détaillées, correctes, cohérentes et concises. Priorisez la précision et la fiabilité.
7. Les articles supplémentaires d'un article sont sur ceux les quels il se refèrent.
6. Réflechissez étape par étape avant de fournir votre réponse finale.
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
        
@server.route("/ocr_system")    
def ocr_system():
    
    try:
        
        return render_template("ocr_system.html", title = "Scan en Texte", page = 'ocr')
            
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
    
    base_n = base_n
    bm25_n = bm25_n
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
    
    # db = rag.get_chroma_vectorstore(documents, embeddings, metric = "l2")
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
    global SEARCH_PROMPT
    global DECISION_PROMPT
    
    print("Entering chat thread")
    
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
        llm_query_rewriter,
        verbose=verbose,
        chat_llm=chat_model,
        logs=logs,
        c_prompt=c_prompt,
        e_prompt=e_prompt,
        target=targets["sel"]
    )
    
    # envoie de la requête
    try:
        result = agent_system.invoke({"query": query}, config = {'recursion_limit': max_iter})
    except RecursionError:
        logs["results"].append([{"log": "Nombre Maximale d'Itérations Atteint !", "color": "red"}])
        logs["nodes"].append("Critère d'Erreur :")
    
    response = result['answer']
    
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
    global SEARCH_PROMPT
    global DECISION_PROMPT
    
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
            if len(result) > log_offset:
                
                if node == "Réponse Finale :":
                    
                    log = result[log_offset]
                    
                    log["log"] = substitute_strong(log["log"].replace("Réponse =>", "").strip())
                    
                    dict_ = {
                        "result": log,
                        "node": node,
                        "query": query,
                        "temperature": temperature,
                        "base_n": base_n,
                        "bm25_n": bm25_n,
                        "max_iter": max_iter,
                        "chat_model": chat_models["sel"],
                        "tr_model": tr_models["sel"],
                        "embedding_id": embedding_ids["sel"],
                        "metric": metrics["sel"],
                        "reranker": rerankers["sel"],
                        "c_prompt": c_prompt, 
                        "e_prompt": e_prompt,
                        "s_prompt": SEARCH_PROMPT,
                        "d_prompt": DECISION_PROMPT
                    }
                    
                    yield f"data: {json.dumps(dict_)}\n\n"
                    
                    break
                
                elif node == "Critère d'Erreur :":
                
                    log = result[log_offset]
                    
                    dict_ = {
                        "result": log,
                        "node": node,
                        "query": query,
                        "temperature": temperature,
                        "base_n": base_n,
                        "bm25_n": bm25_n,
                        "max_iter": max_iter,
                        "chat_model": chat_models["sel"],
                        "tr_model": tr_models["sel"],
                        "embedding_id": embedding_ids["sel"],
                        "metric": metrics["sel"],
                        "reranker": rerankers["sel"],
                        "c_prompt": c_prompt, 
                        "e_prompt": e_prompt,
                        "s_prompt": SEARCH_PROMPT,
                        "d_prompt": DECISION_PROMPT
                    }
                    
                    yield f"data: {json.dumps(dict_)}\n\n"
                    
                    break
            
                if log_offset + 1 < len(result):
                    
                    dict_ = {
                        "result": result[log_offset],
                        "node": node
                    }
                
                    log_offset += 1
                    
                    yield f"data: {json.dumps(dict_)}\n\n"
                    
                elif len(logs["results"]) > result_offset + 1:
                    
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
    global load
    
    # copy metadata dataframe
    articles_ = articles.copy()
    
    # initialize the context
    contexts = []
    
    try:
        
        if request.method == "POST":
            
            # domaine = request.form.get('domaine')
            # loi = request.form.get('loi')
            # decret = request.form.get('decret')
            # arrete = request.form.get('arrete')
            # declaration = request.form.get('declaration')
            # partie = request.form.get('partie')
            # livre = request.form.get('livre')
            # titre = request.form.get('titre')
            # sous_titre = request.form.get('sous_titre')
            # chapitre = request.form.get('chapitre')
            # section = request.form.get('section')
            # sous_section = request.form.get('sous_section')
            # application = request.form.get('application')
            # loyer = request.form.get('loyer')
            # localite = request.form.get('localite')
            # categorie = request.form.get('categorie')
            # habitation = request.form.get('habitation')

            query = request.form.get('query')
            
            base_n = request.form.get('base_n')
            
            base_n = int(base_n)
            
            bm25_n = request.form.get('bm25_n')
            
            bm25_n = int(bm25_n)
            
            temperature = request.form.get('temperature')
            
            temperature = float(temperature)
            
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
            
            # if query is None:
                
                # articles_.rename(column_map, axis = 1, inplace = True)
                
                # articles_ = mix.filter_articles(
                #     articles_,
                #     domaine = domaine,
                #     loi = loi,
                #     decret = decret,
                #     arrete = arrete,
                #     declaration = declaration,
                #     partie = partie,
                #     livre = livre,
                #     titre = titre,
                #     sous_titre = sous_titre,
                #     chapitre = chapitre,
                #     section = section,
                #     sous_section = sous_section,
                #     application = application,
                #     loyer = loyer,
                #     localite = localite,
                #     categorie = categorie,
                #     habitation = habitation
                # )
                  
                # uniques = mix.get_metadata_as_dict(articles_)
                    
                # return jsonify(uniques)
                    
            if query != '' and not query is None:
            # elif query != '':
                
                metadata = {
                    'domaine': domaine,
                    'numero_loi': loi,
                    'numero_decret': decret,
                    'numero_arrete': arrete,
                    'declaration': declaration,
                    'division_partie': partie,
                    'division_livre': livre,
                    'division_titre': titre,
                    'division_sous_titre': sous_titre,
                    'division_chapitre': chapitre,
                    'division_section': section,
                    'division_sous_section': sous_section,
                    'application': application,
                    'loyer': loyer,
                    'localite': localite,
                    'categorie': categorie,
                    'habitation': habitation
                }
                
                uniques = mix.get_metadata_as_dict(articles)
                
                retriever, filters, documents, db, embeddings = rag_generator()

                documents = retriever.invoke(query)

                contexts = [doc.page_content for doc in retriever.invoke(query)]
                
                chat_llm = ChatGroq(model=chat_models["sel"].replace("(groq)", "").strip(), temperature=temperature) if "groq" in chat_models["sel"]\
                    else ChatMistralAI(model=chat_models["sel"].replace("(mistral)", "").strip(), temperature=temperature)
                
                # envoie de la requête
                response = rag.get_answer(
                    chat_llm,
                    query,
                    documents,
                    targets["sel"],
                    c_prompt,
                    e_prompt
                )
                
                
                # data_ = pd.read_csv("Mixtral_8x7B_Instruct_v0.1.csv")
                    
                # data_.loc[len(data_)] = [query.replace('\n', '\\n'), response.replace('\n', '\\n'), k, temperature, {key: value for key, value in metadata.items() if value != ''}]
                
                # data_.to_csv("Mixtral_8x7B_Instruct_v0.1.csv", index = False)
                
            else:
                
                # uniques = mix.get_metadata_as_dict(articles)
                
                response = "Veuillez fournir une requête avant de soumettre !"
            
            return jsonify({"title": "Réponse du système de RAG", "response":substitute_strong(response.replace('\n', '<br>')),
                            "context": [text.replace('\n', '<br>') for text in contexts], "query": query, 
                            "correct": len(contexts) > 0, "result": True, "temperature": temperature, "base_n": base_n, 
                            "bm25_n": bm25_n, "chat_model": chat_models["sel"], 
                            "embedding_id": embedding_ids["sel"], "metric": metrics["sel"],
                            "reranker": rerankers["sel"],
                            "c_prompt": c_prompt, "e_prompt": e_prompt})
        
        else:
            
            # uniques = mix.get_metadata_as_dict(articles)
            return render_template("rag_system.html", result = False, title = "Système de RAG", page = 'rag', base_n = base_n, bm25_n = bm25_n, chat_models = chat_models,
                                   embedding_ids = embedding_ids, metrics = metrics, rerankers = rerankers,
                                   temperature = temperature, c_prompt = c_prompt, e_prompt = e_prompt,
                                #    metadata = uniques
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
    global temperature
    global chat_models
    global tr_models
    global response
    global c_prompt
    global e_prompt
    global SEARCH_PROMPT
    global DECISION_PROMPT
    global load
    
    # recuperate data for selection
    articles_ = articles.copy()
     
    try:
        
        if request.method == "POST":
            
            # domaine = request.form.get('domaine')
            # loi = request.form.get('loi')
            # decret = request.form.get('decret')
            # arrete = request.form.get('arrete')
            # declaration = request.form.get('declaration')
            # partie = request.form.get('partie')
            # livre = request.form.get('livre')
            # titre = request.form.get('titre')
            # sous_titre = request.form.get('sous_titre')
            # chapitre = request.form.get('chapitre')
            # section = request.form.get('section')
            # sous_section = request.form.get('sous_section')
            # application = request.form.get('application')
            # loyer = request.form.get('loyer')
            # localite = request.form.get('localite')
            # categorie = request.form.get('categorie')
            # habitation = request.form.get('habitation')

            query = request.form.get('query')
            
            base_n = request.form.get('base_n')
            
            base_n = int(base_n)
            
            bm25_n = request.form.get('bm25_n')
            
            bm25_n = int(bm25_n)
            
            max_iter = request.form.get('max_iter')
            
            max_iter = int(max_iter)
            
            temperature = request.form.get('temperature')
            
            temperature = float(temperature)
            
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
            
            SEARCH_PROMPT = request.form.get('s_prompt')
            
            DECISION_PROMPT = request.form.get('d_prompt')
            
            # if query is None:
                
                # articles_.rename(column_map, axis = 1, inplace = True)
                
                # articles_ = mix.filter_articles(
                #     articles_,
                #     domaine = domaine,
                #     loi = loi,
                #     decret = decret,
                #     arrete = arrete,
                #     declaration = declaration,
                #     partie = partie,
                #     livre = livre,
                #     titre = titre,
                #     sous_titre = sous_titre,
                #     chapitre = chapitre,
                #     section = section,
                #     sous_section = sous_section,
                #     application = application,
                #     loyer = loyer,
                #     localite = localite,
                #     categorie = categorie,
                #     habitation = habitation
                # )
                
                # uniques = mix.get_metadata_as_dict(articles_)
                    
                # return jsonify(uniques)
                    
            if query != '' and not query is None:
            # elif query != '':
                
                metadata = {
                    'domaine': domaine,
                    'numero_loi': loi,
                    'numero_decret': decret,
                    'numero_arrete': arrete,
                    'declaration': declaration,
                    'division_partie': partie,
                    'division_livre': livre,
                    'division_titre': titre,
                    'division_sous_titre': sous_titre,
                    'division_chapitre': chapitre,
                    'division_section': section,
                    'division_sous_section': sous_section,
                    'application': application,
                    'loyer': loyer,
                    'localite': localite,
                    'categorie': categorie,
                    'habitation': habitation
                }
                
                uniques = mix.get_metadata_as_dict(articles)
                
                # data_ = pd.read_csv("Mixtral_8x7B_Instruct_v0.1.csv")
                    
                # data_.loc[len(data_)] = [query.replace('\n', '\\n'), response.replace('\n', '\\n'), k, temperature, {key: value for key, value in metadata.items() if value != ''}]
                
                # data_.to_csv("Mixtral_8x7B_Instruct_v0.1.csv", index = False)
                
                correct = True
                
            else:
                
                # uniques = mix.get_metadata_as_dict(articles)
                
                response = "Veuillez fournir une requête avant de soumettre !"

                correct = False
            
            return jsonify(
                    {
                        'correct': correct,
                        'result': True,
                        'title': "Réponse obtenue du système d'agent"
                    }
            )
            # return jsonify({
            #     'title': "Réponse obtenue de l'agent appliqué aux vecteurs",
            #     'response': response.replace('\n', '<br>'),
            #     'query': query.replace('\n', '<br>') if not query is None else "",
            #     'metadata': uniques
            # })
        
        else:
            
            # uniques = mix.get_metadata_as_dict(articles)
            return render_template("agent_system.html", result = False, title = "Système Multi-Agent", page = 'agent', base_n = base_n, bm25_n = bm25_n, max_iter = max_iter, 
                                   temperature = temperature, chat_models = chat_models, tr_models = tr_models,
                                   embedding_ids = embedding_ids, metrics = metrics, rerankers = rerankers,
                                   c_prompt = c_prompt, e_prompt = e_prompt,
                                   s_prompt = SEARCH_PROMPT, d_prompt = DECISION_PROMPT,
                                #    metadata = uniques
                                   )
            
    except Exception as e:
        return render_template(
            "error.html",
            title="Error",
            error=e,
            result=False,
            src = "https://www.youtube.com/embed/dsUXAEzaC3Q"
        )



