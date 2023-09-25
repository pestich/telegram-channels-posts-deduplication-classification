# Telegram Channel Post Deduplicator and Classifier

This project is a tool for analyzing and processing textual data from Telegram channels. It consists of two main components: a deduplicator and a text classifier. The project is implemented using various libraries and technologies, including nltk, spacy, datasketch, transformers, and sentence-transformers. The microservice is built using FastAPI.

## Key Features of the Project:

### Data Preprocessing

During the data preprocessing stage, text is cleaned and prepared. There are two main versions of data preprocessing:

1. **With Text Lemmatization:** Texts undergo lemmatization using the spacy library. This step helps reduce data dimensionality and improve analysis accuracy but requires more time.

2. **Without Lemmatization:** Texts are also cleaned but without lemmatization.

### Text Deduplication

Text deduplication consists of two stages:

1. **MinHash and LSH:** At this stage, potential text duplicates are identified. These algorithms allow for quick detection of duplicate candidates, and this approach is easily scalable to process tens of thousands of messages.

2. **Cosine Similarity Evaluation (SBERT):** After obtaining duplicate candidates, texts are checked for cosine similarity using the SBERT (Sentence-BERT) model. This step helps determine whether the texts are indeed duplicates.

Threshold values can be set at each stage, providing flexibility in configuring the deduplication service and striking a balance between quality and algorithm speed.

### Text Classification

The classification task involves categorizing texts into one of 29 predefined classes (topics). Two methods are used for this task:

1. **BERT:** Assigning text to one of the 27 classes.

2. **Rule-Based System:** For specific topics, classification is based on rules associated with Telegram channel identifiers. This allows for precise classification into two additional classes.

## Technologies and Libraries

The project utilizes the following technologies and libraries:

- [nltk](https://www.nltk.org/): A library for natural language processing.
- [spacy](https://spacy.io/): A tool for lemmatization and text processing.
- [datasketch](https://ekzhu.github.io/datasketch/index.html): A library for working with MinHash and LSH algorithms.
- [transformers](https://huggingface.co/transformers/): A library for working with Transformer models.
- [sentence-transformers](https://www.sbert.net/): A library for working with the SBERT model.
- [FastAPI](https://fastapi.tiangolo.com/): A framework for building microservices.
