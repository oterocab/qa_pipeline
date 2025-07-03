# RAG pipeline over a biomedical corpus (BioASQ dataset 2024)

## Introduction

This project was built with the objective of building a functional pipeline 
for information retrieval and answer generation tasks using the BioASQ12 Task B dataset as a foundation.

In order to simulate a "real world" scenario, we have created the required modules and scripts which allow fetching the Pubmed documents using the 
NCBI API, storing them in a structured format into a PostgreSQL table. From this stored documents we are allowed to build multiple tables that
act as vector stores for each embedding model used for dense retrieval (cosine similarity search using PGVector extension).

Within the developed modules we will find utilies for document preprocessing and chunking, interfaces for the embedding and rerankers, that are the 
core of the IR, and also and iterface for the Reader object in charge of anwer synthesis from the provided context using the retrieval tools.

On top of that we have created a simple streamlit app to have a straight forward way of interacting with the QA pipeline, allowing us to set the multiple configurations, making natural language queries, and evaluating the retrieval performance or the end-to-end performance of the system using RAGAS.


This project uses:
- the dataset from BioASQ12 Task B challenge: 5046 labeled questions in the training set. https://participants-area.bioasq.org/datasets/
- `FlagEmbedding` for fine-tuning reranker models: https://github.com/FlagOpen/FlagEmbedding/tree/master
- PostgreSQL to act as a vector store with PGVector extension.
- Python 3.10

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/user/tfm-repo.git

   ```

2. Set up an python environment and install all required dependencies:
    ```bash
    pip install -r requirements.txt
     ```
    If nltk data are not downloaded automatically, make sure to run this in your python environment:
    import nltk

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

3. Clone and install the FlagEmbedding repository for fine tunning and model usage
    ```bash
    git clone https://github.com/FlagOpen/FlagEmbedding.git
    cd FlagEmbedding
    pip install  .[finetune]
     ```

4. Set up the PostgreSQL database and install the PGVector extension
    ```bash
        sudo apt update
        sudo apt install postgresql
        git clone https://github.com/pgvector/pgvector.git
        cd pgvector
        make
        sudo make install
    ```
    After connect to the database that will be used to store the vectors and run:
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
    under examples/SQL there is a SQL script that contains template code to build the 
    corpus table to store all documents and the vector store tables for all the embedding set ups that will be used.

5. Set your ENV file variables. Inside .example.env there a breaktrough of what needs to be set.
   Configuration is centralized using the AppConfig class under utils/app_config.py. Paths and variables can be left as default, or can be overriden with custom ones.




## Usage

To train a model you can provide a configuration file with all the training parameters and run:
```python
    python scripts/fine_tune_pipeline.py --config config/file/path.yaml --steps hn_mine score train (any or all of them can be triggered)
```
 You can find the template config file at config/fine_tune_config/example_fine_tune.yaml

 To use the app you need to provide in the config/retrieval_config all the required object definitions. By default the project comes with a template that can be reused for this purpose with some base, embedding, reranking, and reader models.

to start the Streamlit App run: 
```bash
streamlit run app.py.
```
