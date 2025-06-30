import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from utils.app_config import BaseAppConfig
from evaluation.retrieval_evaluation import evaluate_retriever_from_evaldata
from utils.streamlit_utils import load_css, dict_hash, compute_avg_metrics_per_k, get_component_config
from utils.streamlit_cached_resources import get_reader, get_db_conn, get_retriever, get_ragas_evaluator, cached_retrieve_docs


load_css("css/app_style.css")
load_dotenv()
config = BaseAppConfig()
embedding_config = config.embedder_config
reranker_config = config.reranker_config
reader_config = config.reader_config
evaluator_llm_name = config.evaluator_llm_name
evaluation_files_dir = config.evaluation_files_dir

# UI

st.sidebar.title("Model Selection")

selected_embedder_config = get_component_config(embedding_config, "Embedder")
selected_reranker_config = get_component_config(reranker_config, "Reranker")
selected_reader_config, provider_name = get_component_config(reader_config, "Reader", include_provider=True)
col_left, col_right = st.columns([3, 2])
with col_left:
        st.title("QA Pipeline")
        top_k = st.sidebar.slider("Top-K documents to retrieve", min_value=10, max_value=1000, value=300, step=10)
        rerank_top_k = st.sidebar.slider("Top-K results from rerank", min_value=1, max_value=top_k, value=30, step=1)
        max_context_docs = st.sidebar.slider("Top-K documents to pass to the reader", min_value=1, max_value=25, value=15, step=1)
        use_reranking = st.sidebar.checkbox("Enable Reranking", value=True)

        # Query Input
        query = st.text_input("Ask a question:")

        col1, col2 = st.columns([1, 2]) 

        with col1:
            search_clicked = st.button("Run Search", use_container_width=True)

        with col2:
            with st.expander("System prompt"):
                system_prompt = st.text_area(
                    "Sys Prompt",
                    value="""Answer the following question as precisely and concisely as possible. 
                             Only answer using the provided context.""",
                    key="sys_prompt"
                )

        reader = get_reader(system_prompt, provider_name, selected_reader_config)
        db_conn = get_db_conn(config.conn_config)
        retriever = get_retriever(
            db_conn,
            "documents",
            selected_embedder_config.get("store_table"),
            selected_embedder_config.get("model_config", None),
            selected_reranker_config,
            selected_embedder_config.get("index_config", {}),
            _hash_marker=dict_hash(selected_embedder_config) + dict_hash(selected_reranker_config)
        )

        if search_clicked and query:
            st.session_state.query = query
            st.session_state.top_k = top_k
            st.session_state.rerank_top_k = rerank_top_k
            st.session_state.max_context_docs = max_context_docs
            st.session_state.use_reranking = use_reranking

            with st.spinner("Retrieving documents..."):
                
                docs, db_latency, rerank_latency = cached_retrieve_docs(_retriever=retriever,
                                            query=query,
                                            top_k=top_k,
                                            rerank_top_k=rerank_top_k,
                                            use_reranking=use_reranking,
                                            embedder_config=selected_embedder_config.get("model_config", {}),
                                            reranker_config=selected_reranker_config,
                            )
                start_time = time.time()
                answer = reader.answer(query, docs[:max_context_docs])
                answer_latency = time.time() - start_time
                total_latency = db_latency + rerank_latency + answer_latency
                st.session_state.answer_latency = answer_latency
                st.session_state.db_latency = db_latency
                st.session_state.rerank_latency = rerank_latency
                st.session_state.total_latency = total_latency
                

            # Into the cache
            st.session_state.docs = docs
            st.session_state.answer = answer
            st.session_state.search_ready = True

        # Display the results when done!
        if st.session_state.get("search_ready", False):
            st.markdown(f"**Answer:** {st.session_state.answer}")
            latency_data = {
                            "Stage": ["Retrieval", "Rerank", "Answer", "Total"],
                            "Latency (s)": [
                                round(st.session_state.db_latency, 3),
                                round(st.session_state.rerank_latency, 3),
                                round(st.session_state.answer_latency, 3),
                                round(st.session_state.total_latency, 3)
                            ]
                        }
            if st.session_state.rerank_latency == 0.0:
                latency_data["Stage"].pop(1)
                latency_data["Latency (s)"].pop(1)

            st.table(pd.DataFrame(latency_data))
            
            if st.session_state.docs and st.session_state.use_reranking and 'rerank_score' in st.session_state.docs[0]:
                fig = px.scatter(
                    st.session_state.docs[:st.session_state.max_context_docs],
                    x="score",
                    y="rerank_score",
                    hover_name="id",
                    text="id",
                    labels={"score": "Score", "rerank_score": "Rerank Score"},
                    title="Score vs. Rerank Score"
                )
                fig.update_traces(marker=dict(size=10), textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top Documents")
            docs_to_show = sorted(st.session_state.docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
            for i, doc in enumerate(docs_to_show[:15]):
                st.markdown(f"### Doc {i+1}: `{doc['id']}`")
                st.markdown(f"**Snippet**: {doc['content']}")

                st.caption(f"Score: {doc['score']:.4f}")
                if "rerank_score" in doc:
                    st.caption(f"Rerank Score: {doc['rerank_score']:.4f}")

                st.markdown(f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{doc['id']}/)")
                st.divider()

with col_right:
        st.header("Evaluation")
        eval_mode = st.radio("Evaluation Mode", ["Retrieval Only", "End-to-End (RAGAS)"], horizontal=True)
        eval_folder = evaluation_files_dir
        eval_files = [f for f in os.listdir(eval_folder) if f.endswith(".json")]

        if eval_files:
            selected_eval_file = st.selectbox("Evaluation files", eval_files, key="eval_file")
            ks = st.multiselect("Select K values for evaluation", options=[5, 10, 15, 20, 30, 50, 100], default=[5, 10, 15], key="ks_select")
            overlap_threshold = st.slider("Overlap Threshold", 0.0, 1.0, 0.5, 0.05, key="overlap_threshold")
            if "eval_results" not in st.session_state:
                st.session_state.eval_results = None
                st.session_state.eval_avg = None

            if "ragas_scores" not in st.session_state:
                st.session_state.ragas_scores = None

            if st.button("Run Evaluation", use_container_width=True):
                with st.spinner("Evaluation ongoing..."):
                    eval_path = os.path.join(eval_folder, selected_eval_file)
                    start_time = time.time()

                    if eval_mode == "Retrieval Only":
                        st.session_state.ragas_scores = None
                        eval_df, mean_retrieval_latency, mean_rerank_latency = evaluate_retriever_from_evaldata(
                            retriever=retriever,
                            eval_file_path=eval_path,
                            ks=ks,
                            top_k=top_k,
                            rerank_top_k=rerank_top_k,
                            use_rerank=use_reranking,
                            overlap_threshold=overlap_threshold,
                            fts_search=not bool(selected_embedder_config.get("model_config"))
                        )

                        avg_metrics = compute_avg_metrics_per_k(eval_df)
                        st.session_state.eval_results = eval_df
                        st.session_state.eval_avg = avg_metrics
                        st.session_state.mean_retrieval_latency = mean_retrieval_latency
                        st.session_state.mean_rerank_latency = mean_rerank_latency
                        st.session_state.mean_total_lantecy = mean_retrieval_latency + mean_rerank_latency
                    else:
                        st.session_state.eval_results = None
                        st.session_state.eval_avg = None
                        ragas_evaluator = get_ragas_evaluator(retriever, 
                                                            reader, 
                                                            search_top_k=top_k, 
                                                            llm_top_k=max_context_docs, 
                                                            reranker_top_k=rerank_top_k, 
                                                            use_reranker=use_reranking, 
                                                            fts_search= not bool(selected_embedder_config.get("model_config")))
                        
                        dataset = ragas_evaluator.create_ragas_dataset(eval_path)
                        if not evaluator_llm_name:
                            raise ValueError("Evaluator LLM name is not set. Please specify a model like 'gpt-3.5-turbo'.")
                        ragas_score = ragas_evaluator.run_ragas_evaluation(dataset, evaluator_llm_name)
                        st.session_state.ragas_scores = ragas_score

                    end_time = time.time()
                    st.session_state.eval_latency = end_time - start_time

            if eval_mode == "Retrieval Only" and st.session_state.eval_results is not None:
                st.subheader("Mean Results")
                st.write(f"Evaluation duration {st.session_state.eval_latency:.2f}s")
                latency_data = {
                            "Stage": ["Retrieval", "Rerank", "Total"],
                            "Latency (s)": [
                                round(st.session_state.mean_retrieval_latency, 3),
                                round(st.session_state.mean_rerank_latency, 3),
                                round(st.session_state.mean_total_lantecy, 3)
                            ]
                        }
                if st.session_state.mean_rerank_latency == 0.0:
                    latency_data["Stage"].pop(1)
                    latency_data["Latency (s)"].pop(1)
                st.table(pd.DataFrame(latency_data))

                st.write(st.session_state.eval_avg)

                st.subheader("Metrics per Query")
                df = st.session_state.eval_results
                st.dataframe(df, use_container_width=True)


                st.subheader("Metrics")
                df_melted = st.session_state.eval_avg.reset_index().melt(id_vars='index')
                fig = px.bar(df_melted, x='index', y='value', color='variable', barmode='group', labels={'index': 'Metric', 'variable': 'k'})
                st.plotly_chart(fig, use_container_width=True)

            elif eval_mode == "End-to-End (RAGAS)" and st.session_state.ragas_scores is not None:
                st.subheader("RAGAS Evaluation Results")
                st.caption(f"Evaluation latency: {st.session_state.eval_latency:.2f} seconds")

                st.subheader("Per-question RAGAS Scores")
                df = st.session_state.ragas_scores.to_pandas()
                st.dataframe(df)

                mean_scores = df[["answer_relevancy", "faithfulness", "context_precision"]].mean() # If some of the metrics for one query failed to be calculated, 
                st.subheader("RAGAS Evaluation Summary")                                           # the record is not taken into account for the mean calculations
                cols = st.columns(len(mean_scores))

                for col, (metric, score) in zip(cols, mean_scores.items()):
                    with col:
                        st.metric(metric.replace("_", " ").title(), f"{score:.4f}")
            
            
        else:                
            st.warning("No se encontraron archivos de evaluación.")
        
    