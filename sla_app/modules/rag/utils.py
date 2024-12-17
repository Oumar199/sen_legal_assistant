from sla_app.modules.rag import HuggingFaceEmbeddings, tqdm, pd, RecursiveCharacterTextSplitter, Document, Chroma, LLMChainFilter, EmbeddingsRedundantFilter, BM25Retriever, EnsembleRetriever, MultiQueryRetriever, HuggingFaceCrossEncoder, CrossEncoderReranker, DocumentCompressorPipeline, ContextualCompressionRetriever, RunnableLambda, itemgetter, time, ChatPromptTemplate, StrOutputParser, faiss, FAISS, InMemoryDocstore, os

class RateLimitError(Exception):
    
    def __init__(self, task):
        
        print(f"Error for the task {task} ! The API's limits are maybe obsolete")
        
        self.task = task

def execute_with_count(max_count, task):  
        
    def execution_decorator(func):
        
        def wrapper(*args, **kwargs):
            
            count = 0
            
            while True:
                
                try:
                    
                    value = func(*args, **kwargs)
                    
                    time.sleep(2)
                    
                    return value
                
                except Exception as e:
                    
                    print(e)
                    
                    if count > max_count:
                        
                        raise RateLimitError(task)

                    time.sleep(2)
                    
                    count += 1
            
        return wrapper
    
    return execution_decorator

def get_embedding(id):

    model = HuggingFaceEmbeddings(model_name=id)

    return model

def insert_metadata(document, before=False, add_contextual_container=False):

    contenu = ""

    x = document.metadata

    contextual_container = ""

    if x["code"] in ["Foncier et Domanial"] and add_contextual_container:

        if "numero_decret" in x and x["numero_decret"] != "":

            contextual_container = f" of décret {x['numero_decret']}"

        elif "numero_loi" in x and x["numero_loi"] != "":

            contextual_container = f" of loi {x['numero_loi']}"

        elif "numero_arrete" in x and x["numero_arrete"] != "":

            contextual_container = f" of arrêté ministériel {x['numero_arrete']}"

    if not before:
        contenu += f"""Content of the article number {x['numero_article']}{contextual_container} : {x['contenu']}\n"""

    contenu += f"References : Code juridique = {x['code']}; "
    if "domaine" in x and x["domaine"] != "":
        contenu += f"Domaine = {x['domaine']}; "
    if "date_signature" in x and x["date_signature"] != "":
        contenu += f"Date de signature = {x['date_signature']}; "
    if "numero_arrete" in x and x["numero_arrete"] != "":
        contenu += f"Arrêté ministériel = {x['numero_arrete']}; "
    if "declaration" in x:
        contenu += f"Déclaration universelle des droits de l'homme et du citoyen = {['non', 'oui'][int(x['declaration'])]}; "
    if "numero_loi" in x and x["numero_loi"] != "":
        contenu += f"Loi = {x['numero_loi']}; "
    if "numero_decret" in x and x["numero_decret"] != "":
        contenu += f"Décret = {x['numero_decret']}; "
    if "division_partie" in x and x["division_partie"] != "":
        contenu += f"Partie = {x['division_partie']}; "
    if "division_livre" in x and x["division_livre"] != "":
        contenu += f"Livre = {x['division_livre']}; "
    if "division_titre" in x and x["division_titre"] != "":
        contenu += f"Titre = {x['division_titre']}; "
    if "division_sous_titre" in x and x["division_sous_titre"] != "":
        contenu += f"Sous-Titre = {x['division_sous_titre']}; "
    if "division_chapitre" in x and x["division_chapitre"] != "":
        contenu += f"Chapitre = {x['division_chapitre']}; "
    if "division_section" in x and x["division_section"] != "":
        contenu += f"Section = {x['division_section']}; "
    if "division_sous_section" in x and x["division_sous_section"] != "":
        contenu += f"Sous-Section = {x['division_sous_section']}; "
    if "division_paragraphe" in x and x["division_paragraphe"] != "":
        contenu += f"Paragraphe = {x['division_paragraphe']}; "
    if "section_application" in x and x["section_application"] != "":
        contenu += f"Section d'application = {x['section_application']}; "
    if "loyer" in x:
        contenu += f"Barêmes des prix des loyer = {['non', 'oui'][int(x['loyer'])]}; "
    if "type_terrain" in x and x["type_terrain"] != "":
        contenu += f"Type de terrain = {x['type_terrain']}; "
    if "region" in x and x["region"] != "":
        contenu += f"Région = {x['region']}; "
    if "localite" in x and x["localite"] != "":
        contenu += f"Localité = {x['localite']}; "
    if "categorie" in x and x["categorie"] != "":
        contenu += f"Catégorie de location = {x['categorie']}; "
    if "type_habitation" in x and x["type_habitation"] != "":
        contenu += f"Type d'habitation = {x['type_habitation']}"

    if before:
        contenu += f"""; \nContent of the article number {x['numero_article']}{contextual_container} : {x['contenu']}"""

    document.page_content = contenu.strip()

    return document

def get_documents_from_files(
    src_files: list,
    chunk_sizes: list,
    add_metadata: bool,
    add_metadata_before: bool,
    add_contextual_container: bool = False,
):

    documents = []
    for src_file, chunk_size in tqdm(zip(src_files, chunk_sizes)):
        document = (
            pd.read_csv(src_file).rename(columns={"id_Article": "id"}).fillna(value="")
        )
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"],
            chunk_size=chunk_size,
            chunk_overlap=int(0.3 * chunk_size),
            length_function=len,
            is_separator_regex=False,
        )

        # create documents with metadata
        docs = [
            Document(
                page_content=document.iloc[i]["contenu"],
                metadata={
                    key: str(value) if key != "numero_article" else int(value)
                    for key, value in document.iloc[i].items()
                },
            )
            for i in range(len(document))
        ]

        documents += text_splitter.split_documents(docs)

        if add_metadata:
            documents = [
                insert_metadata(doc, add_metadata_before, add_contextual_container)
                for doc in documents
            ]

    for i in range(len(documents)):

        documents[i].metadata["id"] = str(i + 1)

    return documents

def faiss_vectorstore(
    documents,
    embeddings,
    metric="cosine",
):
    
    if metric == 'l2':
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    elif metric == 'cosine':
        index = faiss.IndexFlatIP(len(embeddings.embed_query("hello world")))
    else:
        raise ValueError("The provided metric is not taken into account!")
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    vector_store.add_documents(documents, ids=[doc.metadata["id"] for doc in documents])

    return vector_store
    
def get_faiss_vectorstore(
    documents,
    embeddings,
    metric="cosine",
    save_file="faiss-index",
    save_local=True,
    load=False
):
    
    if save_local:
        
        if load and os.path.exists(save_file):

            vector_store = FAISS.load_local(
                save_file, embeddings, allow_dangerous_deserialization=True
            )

        else:
            
            vector_store = faiss_vectorstore(documents, embeddings, metric)
            
            vector_store.save_local(save_file)
            
    else:
        
        vector_store = faiss_vectorstore(documents, embeddings, metric)
        
    return  vector_store

def get_chroma_vectorstore(
    documents,
    embeddings,
    collection_name="legal-text",
    metric="cosine",
    directory="./legal-text",
):
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": metric},
        persist_directory=directory,
        ids=[doc.metadata["id"] for doc in documents],
    )

    return db

def get_document_extractor(db, n=4, threshold=0.3, metadata={}):

    search_kwargs = {"k": n, "score_threshold": threshold}

    if len(metadata) != 0: search_kwargs["filter"]= metadata

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )

    return retriever

def get_advanced_retriever(
    filter_llm,
    base_retriever,
    documents,
    embeddings,
    n=3,
    rerank=True,
    reranker="antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR",
    top_rerank=6,
    add_hybrid_search=True,
    multiple_query_llm=None,
    include_original=True,
    bm25_weight=0.7,
    base_weight=0.3,
):

    llm = filter_llm

    #  decides which of the initially retrieved documents to filter out and which ones to return
    _filter = LLMChainFilter.from_llm(llm=llm)

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    transformers = [redundant_filter, _filter]

    if add_hybrid_search:

        bm25_retriever = BM25Retriever.from_documents(documents)

        bm25_retriever.k = n

        # reciprocal rank fusion
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, base_retriever], weights=[bm25_weight, base_weight]
        )

    if not multiple_query_llm is None:

        multi_query_llm = multiple_query_llm

        base_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=multi_query_llm,
            include_original=include_original,
        )

    if rerank:

        reranker = HuggingFaceCrossEncoder(model_name=reranker)

        reranker_compressor = CrossEncoderReranker(model=reranker, top_n=top_rerank)

        transformers.append(reranker_compressor)

    # define a pipeline
    compression_pipeline = DocumentCompressorPipeline(transformers=transformers)

    # retrieves the documents similar to query and then applies the compressor
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compression_pipeline, base_retriever=base_retriever
    )

    return compression_retriever

def get_base_retriever_and_filters(
    filter_llm,
    base_retriever,
    documents,
    embeddings,
    n=3,
    rerank=True,
    reranker="antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR",
    top_rerank=6,
    add_hybrid_search=True,
    multiple_query_llm=None,
    include_original=True,
    bm25_weight=0.7,
    base_weight=0.3,
    metadata={},
):

    llm = filter_llm

    #  decides which of the initially retrieved documents to filter out and which ones to return
    _filter = LLMChainFilter.from_llm(llm=llm)

    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    transformers = [redundant_filter, _filter]

    if add_hybrid_search:

        bm25_retriever = BM25Retriever.from_documents(documents)

        bm25_retriever.k = n
        
        if len(metadata) != 0: bm25_retriever.metadata = metadata

        # reciprocal rank fusion
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, base_retriever],
            weights=[bm25_weight, base_weight],
        )

    if not multiple_query_llm is None:

        multi_query_llm = multiple_query_llm

        base_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=multi_query_llm,
            include_original=include_original,
        )

    if rerank:

        reranker = HuggingFaceCrossEncoder(model_name=reranker)

        reranker_compressor = CrossEncoderReranker(model=reranker, top_n=top_rerank)

        transformers.append(reranker_compressor)

    # define a pipeline
    compression_pipeline = DocumentCompressorPipeline(transformers=transformers)

    return base_retriever, compression_pipeline

def format_docs(docs):  # function to format documents
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(
    chat_llm,
    query,
    documents,
    target = "citizen",
    c_prompt = None,
    e_prompt = None,
    max_retries = 10,
):

    qa_rag_chain = get_rag_chain(chat_llm, target=target, c_prompt=c_prompt, e_prompt=e_prompt)
    
    @execute_with_count(max_retries, "answer")
    def execute():
        # send query
        answer = qa_rag_chain.invoke(
            {"context": documents, "question": query}
        )
        
        return answer

    return execute()

def get_rag_chain(chat_llm, target="citizen", c_prompt = None, e_prompt = None):

    c_prompt = """You are a question-answering assistant for Senegalese law, intended for Senegalese citizens. Use simple French. Respond only using the provided context. If no relevant information is available, clearly indicate that you do not know the answer. Do not create unfounded or speculative answers.

1. Use of context: Respond exclusively based on the elements provided in the context. Do not supplement with external information or assumptions.
2. Precise references: Include only exact references (laws or decrees) without mentioning chapters or titles. If a reference is not provided, do not invent one.
3. Verification: Ensure that all sentences are correct and that the references are precise. Avoid ambiguities.
4. Clarity and precision: Provide detailed, correct, coherent, and concise answers. Leave no room for interpretation.
5. Think step-by-step before providing your final answer.
6. Additional articles of an article refer to those specified.""" if c_prompt is None else c_prompt
    
    e_prompt = """You are a question-answering assistant for experts in Senegalese law. Use language suitable for professionals and provide all answers in French. Respond only using the provided context.

1. Use of context: Respond exclusively based on the provided elements. Do not supplement with external information or assumptions.
2. Precise references: Include only exact references (laws or decrees) without mentioning chapters or titles. If a reference is not present, do not invent one.
3. Clarity and precision: Ensure that all sentences are correct and that the references are precise. Avoid ambiguities.
4. Incomplete answers: If no relevant information is available in the context, clearly indicate that you do not know the answer. Do not fabricate answers.
5. Details and completeness: Provide detailed, correct, coherent, and concise answers. Prioritize accuracy and reliability.
6. Think step-by-step before providing your final answer.
7. Additional articles of an article refer to those specified.""" if e_prompt is None else e_prompt
    
    citizen_prompt = c_prompt + """
    
Question :
{question}

Context :
{context}

Answer :
"""

    expert_prompt = e_prompt + """
        
Question :
{question}

Context :
{context}

Answer :
"""

    if target == "citizen":

        prompt_template = ChatPromptTemplate.from_template(citizen_prompt)

    else:

        prompt_template = ChatPromptTemplate.from_template(expert_prompt)

    # create QA RAG chain
    qa_rag_chain = (
        {
            "context": (itemgetter("context") | RunnableLambda(format_docs)),
            "question": itemgetter("question"),
        }
        | prompt_template
        | chat_llm
        | StrOutputParser()
    )

    return qa_rag_chain