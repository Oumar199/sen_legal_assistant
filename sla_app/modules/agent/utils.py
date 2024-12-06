from sla_app.modules.agent import re, ChatMistralAI, TavilySearchResults, DuckDuckGoSearchRun, StateGraph, END, TypedDict, List, Document, Field, BaseModel, time, tqdm, ChatPromptTemplate, PromptTemplate, StrOutputParser, KG_TRIPLE_DELIMITER, ChatGroq
from sla_app.modules.rag.utils import get_rag_chain

# Basic ANSI escape codes for colors
class TextColors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

def remove_metadata(docs, metadata_start="Références"):

    contents = []

    for doc in docs:

        if isinstance(doc, Document):

            contents.append(
                "\n".join(
                    [
                        content
                        for content in doc.page_content.split("\n")
                        if not content.startswith(metadata_start)
                    ]
                )
            )

        else:

            contents.append(
                "\n".join(
                    [
                        content
                        for content in doc.split("\n")
                        if not content.startswith(metadata_start)
                    ]
                )
            )

    return contents

def retrieve_content(docs, metadata_start="Références"):

    contents = [
        ":".join(content.split(":")[1:])
        for content in remove_metadata(docs, metadata_start)
    ]

    return contents

def triple_matching(triples_list, expression):

    matching_triples = []

    for triple in triples_list:

        if isinstance(triple, dict):

            index = triple["index"]

            triple = triple["triple"]

            if re.search(expression, triple):

                matching_triples.append({"index": index, "triple": triple})

        elif isinstance(triple, str):

            if re.search(expression, triple):

                matching_triples.append(triple)

    return matching_triples

def get_tavily_search_tool(n=4, advanced=False, **kwargs):

    if advanced:

        search_tool = TavilySearchResults(
            max_results=n, search_depth="advanced", **kwargs
        )  # advanced web search
    else:

        search_tool = TavilySearchResults(max_results=n, **kwargs)  # simple web search

    return search_tool


def get_ddgo_search_tool(n=4, advanced=False, **kwargs):

    if advanced:

        search_tool = DuckDuckGoSearchRun(
            max_results=n, search_depth="advanced", **kwargs
        )  # advanced web search
    else:

        search_tool = DuckDuckGoSearchRun(max_results=n, **kwargs)  # simple web search

    return search_tool

def retrieve_documents(
    base_retriever, query, compression_pipeline=None, return_filter_percentage=False
):

    documents = base_retriever.invoke(query)

    n_docs = len(documents)

    if not compression_pipeline is None:

        while True:

            try:

                filtered_docs = compression_pipeline.compress_documents(
                    documents, query
                )

                break

            except IndexError:

                filtered_docs = []

                break
            
            except Exception:

                time.sleep(2)

        n_filtered = len(filtered_docs)

        if return_filter_percentage:

            return filtered_docs, n_filtered / n_docs

        else:

            return filtered_docs

    else:

        return documents

def col_print(text, color):
    print(f"{color}{text}{TextColors.RESET}")
                
def add_extracted_ref_to_docs(ref_documents, references):

    ref_docs_and_extractions = []

    for i, ref in tqdm(enumerate(references)):

        ref_extraction = ", ".join([f"{c} = {v}" for c, v in ref.dict().items()])

        ref_document = (
            ref_documents[i]
            if not isinstance(ref_documents[i], Document)
            else ref_documents[i].page_content
        )

        ref_docs_and_extractions.append(
            ref_document + f"\nRéférences extraites : {ref_extraction}"
        )

    return ref_docs_and_extractions

def get_triples_open(llm, prompt_template):

    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template,
    )

    # Create an LLMChain using the knowledge triple extraction prompt
    chain = prompt | llm | StrOutputParser()

    def send_text(text):

        triples = chain.invoke(text)

        return re.sub(r"\bEND\w*(?:\s+\w*)*(?<!\()\b", "", triples).strip()
        # return triples.split("Output:")[-1].split("END OF EXAMPLE")[0].strip("\nEND").strip()

    return send_text

class MakeDecision(BaseModel):
    """Décision entre générer une réponse finale, rechercher plus d'informations sur le web ou récupérer plus de documents selon les références dans le contexte."""

    decision: str = Field(
        description=(
            " Si le nombre de documents de références est strictement supérieur à 0, et si le nombre de tentative d'extraction de références ne dépasse pas 3, et si le contexte spécifié avant le contexte extrait du web, contient une quelconque référence à des articles, lois, décrets, arrêté, Livres de texte juridique, Titres de texte juridique, Chapitres de texte juridique, Sections de texte juridique, Sous-sections de texte juridique ou Paragraphes, alors renvoyer 'references'."
            " Sinon si le contexte ne contient aucune référence et n'est pas suffisamment clair, si le pourcentage de filtrage dépasse 70% et que le nombre de tentative de recherche ne dépasse pas 3, alors renvoyer 'search'"
            " Sinon si le contexte contient suffisamment d'informations pour répondre à la requête, renvoyer 'final_answer'."
        )
    )

def get_mistral_llm_for_decision(model_name):
    # LLM for making decision
    model = ChatMistralAI(model=model_name)
    # Assign structured_llm_decider to model.with_structured_output instead of llm.with_structured_output
    structured_llm_decider = model.with_structured_output(MakeDecision)

    return structured_llm_decider  # return structured_llm_decider


def get_groq_llm_for_decision(model_name):
    # LLM for making decision
    model = ChatGroq(model=model_name)
    # Assign structured_llm_decider to model.with_structured_output instead of llm.with_structured_output
    structured_llm_decider = model.with_structured_output(MakeDecision)

    return structured_llm_decider  # return structured_llm_decider

def get_decider(llm, sys_prompt=None, human_prompt=None):

    # Modèle de prompt pour la vérification des références
    SYS_PROMPT = (
        """Vous êtes un expert en prise décision en fonction d'un contexte obtenu à partir d'une requête.
Suivez ces instructions pour décider :
- Réfléchissez étape par étape pour fournir votre décision.
- Si le nombre de documents de références est strictement supérieur à 0, si le nombre de tentative d'extraction de références ne dépasse pas 3, et le contexte avant le contexte extrait du web contient une quelconque référence à des articles juridiques, lois, décrets, arrêté, Livres de texte juridique, Titres de texte juridique, Chapitres de texte juridique, Sections de texte juridique, Sous-sections de texte juridique ou Paragraphes, alors retourner 'references'.
- Sinon si le contexte ne contient aucune référence et n'est pas assez clair ou si le pourcentage de filtrage dépasse 70% et que le nombre de tentative de recherche ne dépasse pas 3, retourner 'search'.
- Sinon si le contexte ne contient aucune référence mais contient suffisamment d'informations pour répondre à la requête, retourner 'final_answer.
"""
        if sys_prompt is None
        else sys_prompt
    )
    HUMAN_PROMPT = (
        """Requête :
                    {query}

                    Contexte :
                    {context}
                  """
        if human_prompt is None
        else human_prompt
    )
    dec_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    # Construire le décideur
    decider = dec_prompt | llm

    return decider

def get_reference_retriever(ref_llm, parser, template=None):

    template = (
        """
  Extract all references from the context for which the contents are not specified.
  {format_instructions}
  context={context}
  """
        if template is None
        else template
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | ref_llm | parser

    return chain

def get_triples_prompt_from_file(path):

    with open(path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def get_web_search_query(llm, sys_prompt=None, human_prompt=None):

    # Prompt template for rewriting
    SYS_PROMPT = (
        """Agissez en tant que reformulateur de questions et effectuez la tâche suivante :
- Transformez la question d'entrée suivante en une version améliorée, optimisée pour la recherche sur le web.
- Lors de la reformulation, examinez la question d'entrée et essayez de raisonner sur l'intention sémantique sous-jacente / la signification.
- La question de sortie doit être rédigée en français.
- Fournissez uniquement la question de sortie et rien d'autre.
"""
        if sys_prompt is None
        else sys_prompt
    )
    HUMAN_PROMPT = (
        """Here is the initial question:
                    {question}

                    Formulate an improved question.
                    """
        if human_prompt is None
        else human_prompt
    )
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            (
                "human",
                HUMAN_PROMPT,
            ),
        ]
    )
    # Create rephraser chain
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    return question_rewriter

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    
    # replace articles by article
    response = response.replace("articles", "article")

    # Regex pattern to match "article L.xx" or "article R.xx"
    pattern = r'\barticle\s+[LR]\.\s*(\d+)'

    # Substitute all matches with "article <number>"
    result = re.sub(pattern, r'article \1', response)

    # Regex pattern to match all relevant lines including the closing parenthesis
    parentheses_pattern = r'\(.*?\)'

    # If you want to find text within parentheses after the substitution
    matches = re.findall(parentheses_pattern, result)

    return [
        element.strip()
        for element in matches
    ]

def web_search(query, search_tool):

    while True:
        
        try:
            docs = search_tool.invoke(query)
            time.sleep(2)
            
            break
        except Exception as e:
            time.sleep(2)

    web_results = "\n\n".join([d["content"] for d in docs])

    web_results = Document(page_content=web_results)

    return web_results

"""Let us define the agent systems using langgraph
"""

# Modèle de données pour le format de sortie des LLM
class References(BaseModel):
    article: List[str] = Field(description="Articles référés dans le document")
    loi: List[str] = Field(description="Lois référées dans le document")
    code: List[str] = Field(description="Codes juridiques référés dans le document")
    decret: List[str] = Field(description="Décrets référés dans le document")
    arrete: List[str] = Field(description="Arrêtés référées dans le document")
    declaration: List[str] = Field(description="Déclarations référées dans le document")
    partie: List[str] = Field(
        description="Parties de texte juridique référées dans le document"
    )
    livre: List[str] = Field(
        description="Livres de texte juridique référés dans le document"
    )
    titre: List[str] = Field(
        description="Titres de texte juridique référés dans le document"
    )
    chapitre: List[str] = Field(
        description="Chapitres de texte juridique référées dans le document"
    )
    section: List[str] = Field(
        description="Sections de texte juridique référées dans le document"
    )
    sous_section: List[str] = Field(
        description="Sous-sections de texte juridique référées dans le document"
    )
    paragraphe: List[str] = Field(
        description="Paragraphes de texte juridique référés dans le document"
    )
    alinea: List[str] = Field(
        description="Alinéas d'articles de texte juridique référés dans le document"
    )  # added

class AgentStatev1(TypedDict):
    # messages: Annotated[list[AnyMessage], operator.add]
    query: str
    filter_percentage: float
    ref_documents: List[Document]
    references_: List[References]
    web_context: Document
    triples: str
    verbose: bool
    decision: bool
    documents: List[Document]
    answer: str
    n_searches: int
    n_ref_retrieves: int

class AgentSystemv1:

    infos = {
        "query": "",
        "web_query": "",
        "context": "",
        "web_context": "",
        "node": "",
        "answer": "",
        "decision": ""
    }
    
    ref_match_pattern = r"(fait\s+référence\s+à,\s*article\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+de\s+(la\s+)*loi\s+\d+[-\d]*|fait\s+référence\s+à,\s*article\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+du\s+décret\s+\d+[-\d]*|fait\s+référence\s+à,\s*alinéa\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+article\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+de\s+(la\s+)*loi\s+\d+[-\d]*|fait\s+référence\s+à,\s*alinéa\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+article\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*\s+du\s+décret\s+\d+[-\d]*)"
    search_tool = get_tavily_search_tool(advanced=True)
    chat_llm = ChatMistralAI(model="open-mixtral-8x22b")

    def __init__(
        self,
        documents,
        embeddings,
        database,
        retriever,
        filters,
        decision_llm,
        ref_retriever,
        triples_llm,
        triple_retriever_prompt,
        query_rewriter_web_search,
        search_tool=None,
        chat_llm=None,
        target="citizen",
        ref_match_pattern=None,
        verbose=False,
        logs=None,
        c_prompt=None,
        e_prompt=None,
    ):

        graph = StateGraph(AgentStatev1)

        self.documents = documents

        self.embeddings = embeddings

        self.database = database

        self.retriever = retriever

        self.filters = filters

        self.decider = decision_llm

        self.ref_retriever = ref_retriever

        self.triples_llm = triples_llm

        self.triple_retriever_prompt = triple_retriever_prompt

        self.query_rewriter_web_search = query_rewriter_web_search
        
        self.log = logs
        
        self.c_prompt = c_prompt
        
        self.e_prompt = e_prompt

        if not search_tool is None:
            self.search_tool = search_tool

        if not chat_llm is None:
            self.chat_llm = chat_llm

        self.target = target

        self.qa_rag_chain = get_rag_chain(self.chat_llm, target=self.target, c_prompt=self.c_prompt, e_prompt=self.e_prompt)

        if not ref_match_pattern is None:
            self.ref_match_pattern = ref_match_pattern

        self.verbose = verbose

        graph.add_node("filter", self.filter_node)

        graph.add_node("get_references", self.extract_references_node)

        graph.add_node("get_triples", self.extract_triples_node)

        graph.add_node("get_reference_articles", self.extract_reference_articles_node)

        graph.add_node("web_search", self.web_search_node)

        graph.add_node("get_final_answer", self.answer_node)

        graph.set_entry_point("filter")

        graph.add_conditional_edges(
            "filter",
            self.decision_node,
            {
                "final_answer": "get_final_answer",
                "search": "web_search",
                "references": "get_references",
            },
        )

        graph.add_conditional_edges(
            "get_reference_articles",
            self.decision_node,
            {
                "final_answer": "get_final_answer",
                "search": "web_search",
                "references": "get_references",
            },
        )

        graph.add_conditional_edges(
            "web_search",
            self.decision_node,
            {
                "final_answer": "get_final_answer",
                "search": "web_search",
                "references": "get_references",
            },
        )

        graph.add_edge("get_references", "get_triples")

        graph.add_edge("get_triples", "get_reference_articles")

        graph.add_edge("get_final_answer", END)

        self.graph = graph.compile()
        
        # logs["new_log"]
    
    def init_log(self, node):
        
        if not self.log is None: 
            self.log['results'].append([])
            self.log["nodes"].append(node)
        
    def update_log(self, log, color):
        
        if not self.log is None: 
            
            self.log["results"][-1].append({"log": log.replace("\n", "<br>"), "color": color})
    
    def filter_node(self, state):
        
        col_print("Entering Advanced Filtering Node:", TextColors.GREEN)
        
        node = "Filtration de Documents :"
        
        self.init_log(node)
        
        query = state["query"]

        docs, perc = retrieve_documents(self.retriever, query, self.filters, True)

        perc = (1 - perc) * 100

        contexts = [doc.page_content for doc in docs]

        context = "\n\n".join([doc for doc in contexts]) if len(contexts) > 0 else "Pas de contexte !"

        if self.verbose:
            
            result = f"""Pourcentage de filtrage => {round(perc, 2)}%
Nombre de documents => {len(docs)}
Documents => {context}"""
#             result = f"""Result:
# filter percentage => {round(perc, 2)}%
# number of documents => {len(docs)}
# documents => {context}"""
            
            col_print(
                "Résultats:\n" + result,
                TextColors.CYAN,
            )
            
            self.update_log(result, "#64f38c")

        col_print("NODE IS ENDED", TextColors.RED)

        state["documents"] = [doc.copy() for doc in docs]
        state["filter_percentage"] = perc
        state["ref_documents"] = docs
        
        time.sleep(2)

        return state

    def decision_node(self, state):

        col_print("Entering Decision Node:", TextColors.GREEN)

        node = "Prise de Décision :"
        
        self.init_log(node)

        web_context = (
            f"\n\nContexte Web : {state['web_context'].page_content}"
            if not state["web_context"] is None
            else ""
        )

        filter_percentage = (
            f"\n\nPourcentage de filtrage : {round(state['filter_percentage'], 2)}%"
            if not state["filter_percentage"] is None
            else ""
        )

        nombre_references = (
            f"\n\nNombre de documents de références : {len(state['ref_documents'])}"
            if not state["ref_documents"] is None
            else f"\n\nNombre de documents de références : 0"
        )

        if state["n_searches"] is None:
            state["n_searches"] = 0

        if state["n_ref_retrieves"] is None:
            state["n_ref_retrieves"] = 0

        n_searches = f"\n\nNombre de tentatives de recherche : {state['n_searches']}"

        n_ref_retrieves = f"\n\nNombre de tentatives d'extraction de références : {state['n_ref_retrieves']}"

        context = (
            "\n\n".join([doc for doc in retrieve_content(state["documents"])])
            + web_context
            + filter_percentage
            + "\n\n---------------------------------------------"
            + nombre_references
            + n_searches
            + n_ref_retrieves
        )

        while True:
            try:
                decision = self.decider.invoke(
                    {"context": context, "query": state["query"]}
                )
                time.sleep(2)
                break
            except Exception as e:
                time.sleep(2)

        decision = decision.decision

        if self.verbose:
            
            result = f"""Contexte => {context}
Décision => {decision}"""
#             result = f"""Result:
# context => {context}
# decision => {decision}"""
            
            col_print(
                "Résultats :\n" + result,
                TextColors.CYAN,
            )
            
            self.update_log(result, "#64f38c")

        if decision == "final_answer":

            text = "Le contexte contient suffisamment d'informations pour fournir la réponse finale."

        elif decision == "search":

            text = "Le contexte ne contient pas assez d'informations pour fournir la réponse finale. Plus d'informations seront récupérées à partir du web."
       
        elif decision == "references":

            text = "Le contexte ne contient pas assez d'informations pour fournir la réponse finale. Plus d'informations seront récupérées à partir des références."

        else:

            raise ValueError(f"Décision non valide: {decision}")

        col_print(
            text,
            TextColors.YELLOW,
        )
        
        self.update_log(text, "yellow")
        
        col_print("NODE IS ENDED", TextColors.RED)

        return decision

    def extract_references_node(self, state):

        col_print("Entering References Extraction Node:", TextColors.GREEN)

        node = "Extraction de Références :"
        
        self.init_log(node)
        
        documents = state["ref_documents"]

        if self.verbose:

            col_print("Result:", TextColors.CYAN)
        
        documents = retrieve_content(documents)
        
        references = []
        
        if len(documents) == 0:
            
            if self.verbose and not self.log is None:
                
                self.update_log("Aucun document n'est disponible pour extraction !", "#f7797d")
        
        for i in range(len(documents)):  # avec filtrage
            
            document = documents[i]
            
            while True:
                try:
                    ref = self.ref_retriever.invoke({"context": document})
                    
                    if self.verbose:
                        
                        log = f"Références extraites du document {i+1} => {ref}"
                        # log = f"References for document {i+1} => {ref}"
                        
                        col_print(log, TextColors.CYAN)
                        
                        self.update_log(log, "#f7b733")
                        
                    time.sleep(2)
                    
                    references.append(ref)
                    
                    break
                
                except Exception as e:
                    time.sleep(2)

        col_print("NODE IS ENDED", TextColors.RED)

        state["references_"] = references

        state["filter_percentage"] = None

        if state["n_ref_retrieves"] is None:
            state["n_ref_retrieves"] = 0

        state["n_searches"] = 0

        state["n_ref_retrieves"] += 1

        return state

    def extract_triples_node(self, state):

        col_print("Entering Triples Extraction Node:", TextColors.GREEN)
        
        node = "Extraction de Triples :"
        
        self.init_log(node)

        ref_documents = []

        references = []

        docs = state["ref_documents"]
        
        result = ""

        for i, ref in enumerate(state["references_"]):

            if not all([el == [] for el in ref.dict().values()]):

                ref_documents.append(docs[i])

                references.append(ref)

        ref_docs_and_extractions = add_extracted_ref_to_docs(ref_documents, references)

        if self.verbose:

            col_print("Result:", TextColors.CYAN)
        
        triples_set = []
        
        if len(ref_docs_and_extractions) == 0:
            
            if self.verbose and not self.log is None:
                
                self.update_log("Pas de triples extractables !", "#f7797d")
        
        for i, ref_doc in enumerate(ref_docs_and_extractions):
            get_triples = get_triples_open(self.triples_llm, self.triple_retriever_prompt)

            while True:
                try:
                    triples = get_triples(ref_doc)
                    time.sleep(2)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(2)

            if self.verbose:
                col_print(
                    f"""triples of document {i+1} =>
    Document -> {ref_doc}
    Triples -> {triples}""",
                    TextColors.CYAN,
                )
                print("-----------")
                
                if not self.log is None: 
                    
                    self.update_log(f"Triples du document {i+1} =>", "#00F260")
                    self.update_log(f"Document -> {ref_doc}", "#64f38c")
                    self.update_log(f"Triples -> {triples}", "#0575E6")
                    self.update_log(f"-----------", "white")
                    # self.update_log(f"Triples of document {i+1} =>", "")
                    # self.update_log(f"Document -> {ref_doc}", "")
                    # self.update_log(f"Triples -> {triples} =>", "")
                    # self.update_log(f"-----------", "")

            triples_set.append(triples)

        col_print("NODE IS ENDED", TextColors.RED)

        state["triples"] = triples_set

        state["ref_documents"] = ref_documents

        state["references_"] = references

        return state

    def extract_reference_articles_node(self, state):

        col_print("Entering Reference Documents Extraction Node:", TextColors.GREEN)

        node = "Extraction de Documents Référencés :"
        
        self.init_log(node)
        
        document_indices = [doc.metadata["id"] for doc in state["documents"]]

        triples = state["triples"]

        triples_list = []

        doc_index_article = set()

        for i in range(len(triples)):

            triples_list.extend(
                [{"index": i, "triple": triple} for triple in parse_triples(triples[i])]
            )

        matching_triples = triple_matching(triples_list, self.ref_match_pattern)

        all_ref_documents = []

        index_change = False

        first_index = None

        for triple in matching_triples:

            index = triple["index"]

            if first_index is None:

                first_index = index

                index_change = True

            elif index == first_index + 1:

                index_change = True

                first_index = index

            else:

                index_change = False

            triple = triple["triple"]

            triple = triple.split(",", 1)[1]

            article_catch = re.search("article\s+\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*", triple)

            decret_catch = re.search("décret\s+\d+[-\d]+", triple)

            loi_catch = re.search("loi\s+\d+[-\d]+", triple)

            ranges = []

            ref_documents = []

            if article_catch:

                article_catch = article_catch[0]

                article_numbers = re.search("\d+(?:\s*,\s*\d*|(?:\s*\.\.\.\s*))*", article_catch)

                article_numbers = [number.strip() for number in article_numbers[0].split(",") if number.strip() != ""]

                for i in range(len(article_numbers)):

                    article_number = article_numbers[i]

                    if (
                        article_number == "..."
                        and i < len(article_numbers) - 1
                        and i > 0
                    ):

                        ranges.extend(
                            [
                                str(j)
                                for j in range(
                                    int(article_numbers[i - 1]) + 1,
                                    int(article_numbers[i + 1]),
                                )
                            ]
                        )

                    elif article_number == "..." and i == 0 and len(article_number) > 1:

                        ranges.append(f"<{article_numbers[i + 1]}")

                    elif article_number == "..." and i == len(article_numbers) - 1:

                        ranges.append(f">{article_numbers[i - 1]}")

                    else:

                        ranges.append(article_number)

            decret = (
                re.search("\d+[-\d]+", decret_catch[0])[0] if decret_catch else None
            )

            loi = re.search("\d+[-\d]+", loi_catch[0])[0] if loi_catch else None

            numbered_ranges = [
                int(range) for range in ranges if not "<" in range and not ">" in range
            ]

            not_numbered_ranges = [
                range for range in ranges if "<" in range or ">" in range
            ]

            for doc in self.documents:

                if not decret is None:

                    if doc.metadata["numero_decret"] == decret:

                        if doc.metadata["numero_article"] in numbered_ranges:

                            ref_documents.append(doc.copy())

                        for not_numbered_range in not_numbered_ranges:

                            if not_numbered_range.startswith("<"):

                                if int(doc.metadata["numero_article"]) < int(
                                    not_numbered_range[1:]
                                ):

                                    ref_documents.append(doc.copy())

                            elif not_numbered_range.endswith(">"):

                                if int(doc.metadata["numero_article"]) > int(
                                    not_numbered_range[1:]
                                ):

                                    ref_documents.append(doc.copy())

                elif not loi is None:

                    if (
                        doc.metadata["numero_loi"] == loi
                        and doc.metadata["numero_decret"] == ""
                    ):

                        if doc.metadata["numero_article"] in numbered_ranges:

                            ref_documents.append(doc.copy())

                        for not_numbered_range in not_numbered_ranges:

                            if not_numbered_range.startswith("<"):

                                if int(doc.metadata["numero_article"]) <= int(
                                    not_numbered_range[1:]
                                ):

                                    ref_documents.append(doc.copy())

                            elif not_numbered_range.endswith(">"):

                                if int(doc.metadata["numero_article"]) >= int(
                                    not_numbered_range[1:]
                                ):

                                    ref_documents.append(doc.copy())

            ref_document = state["ref_documents"][index]

            reference = state["references_"][index]

            # duplicated = []

            # for i in range(len(state["documents"])):

            #   for j in range(len(ref_documents)):

            #     if state["documents"][i].page_content == ref_documents[j].page_content:

            #       duplicated.append(ref_documents[j])

            doc_index = ref_document.metadata["id"]

            for article in reference.dict()["article"]:

                for i in range(len(state["documents"])):

                    if (
                        state["documents"][i].metadata["id"] == doc_index
                        and not (article, doc_index) in doc_index_article
                    ):

                        state["documents"][i].page_content = re.compile(
                            re.escape(str.lower(article)), re.IGNORECASE
                        ).sub(
                            f"{article} (voir Article(s) supplémentaire(s))",
                            state["documents"][i].page_content,
                        )

                        doc_index_article.add((article, doc_index))

            for j in range(len(ref_documents)):

                ref_document_no_metadata = remove_metadata([ref_documents[j]])[0]

                if index_change:

                    added = f"\nArticle(s) supplémentaire(s) :\n-> {ref_document_no_metadata}"

                    distance = 0

                else:

                    added = f"\n-> {ref_document_no_metadata}"

                    distance += 1

                for i in range(len(state["documents"])):

                    if state["documents"][i].metadata["id"] == doc_index:

                        state["documents"][i].page_content += added

                if ref_documents[j].metadata["id"] not in document_indices:

                    all_ref_documents.append(ref_documents[j])

                    for i in range(len(state["documents"])):

                        if state["documents"][i].metadata["id"] == doc_index:

                            state["documents"].insert(
                                i + distance + 1, ref_documents[j]
                            )

        context = "\n\n".join([doc.page_content for doc in all_ref_documents])

        if self.verbose:
            result = f"""Nombre de documents => {len(all_ref_documents)}
Documents de référence => {context}"""
            col_print(
                "Résultats:\n" + result,
                TextColors.CYAN,
            )
            
#             result = f"""Result:
# number of documents => {len(all_ref_documents)}
# reference documents => {context}"""
#             col_print(
#                 result,
#                 TextColors.CYAN,
#             )
            
            self.update_log(result, "#64f38c")

        col_print("NODE IS ENDED", TextColors.RED)

        state["ref_documents"] = all_ref_documents
        
        time.sleep(2)

        return state

    def web_search_node(self, state):

        col_print("Entering Web Search Node:", TextColors.GREEN)
 
        node = "Extraction de contexte à partir du Web :"
        
        self.init_log(node)

        while True:

            try:

                new_query = self.query_rewriter_web_search.invoke(state["query"])
                
                time.sleep(2)

                break

            except Exception as e:

                time.sleep(2)

        web_context = web_search(new_query, self.search_tool)

        web_context.page_content = web_context.page_content.replace("\n\n", "\n")

        if not state["web_context"] is None:

            state["web_context"].page_content += f"\n{web_context.page_content}"

        else:

            state["web_context"] = web_context

        if self.verbose:

            result = f"""Nouvelle requête => {new_query}
Contexte web => {web_context.page_content}"""
#             result = f"""Result:
# New query => {new_query}
# Web context => {web_context.page_content}"""
            
            col_print("Résultats:\n" + result, TextColors.CYAN)

            self.update_log(result, "#64f38c")
            
        col_print("NODE IS ENDED", TextColors.RED)

        state["filter_percentage"] = None

        if state["n_searches"] is None:
            state["n_searches"] = 0

        state["n_ref_retrieves"] = 0

        state["n_searches"] += 1

        return state

    def answer_node(self, state):

        col_print("Entering Answer Node:", TextColors.GREEN)
        
        node = "Réponse Finale :"
        
        self.init_log(node)

        documents = (
            state["documents"] + [state["web_context"]]
            if not state["web_context"] is None
            else state["documents"]
        )

        while True:

            try:

                answer = self.qa_rag_chain.invoke(
                    {"context": documents, "question": state["query"]}
                )
                
                time.sleep(2)

                break

            except Exception as e:
                
                print(e)

                time.sleep(2)

        # if self.verbose:
        
        result = f"""Réponse => {answer}"""
#         result = f"""Result:
# Answer => {answer}"""
            
        col_print("Résultats:\n" + result, TextColors.CYAN)
        
        self.update_log(result, "#6dd5ed")

        col_print("NODE IS ENDED", TextColors.RED)

        state["answer"] = answer

        return state

    def invoke(self, state, **kwargs):

        col_print(f"User query => {state['query']}", TextColors.BLUE)

        return self.graph.invoke(state, **kwargs)