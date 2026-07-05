# Persian Text Analysis

Persian text analysis web app using TF-IDF, Word2Vec, and Sentence Transformers.
Includes SVD dimensionality reduction, cosine similarity, clustering,
and method comparison — built with Streamlit.

## Features
- Persian text preprocessing with Hazm
- Text vectorization: TF-IDF, Word2Vec, Sentence Transformer
- SVD dimensionality reduction
- Cosine similarity heatmap
- Hierarchical clustering (Dendrogram)
- Side-by-side method comparison

## Which method is better?
| Method | Strength |
|---|---|
| TF-IDF | Fast, interpretable, good for keyword matching |
| Word2Vec | Captures word-level semantics |
| Sentence Transformer | Best for sentence-level semantic similarity |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run text_analysis.py

## Requirements
- streamlit, numpy, pandas, matplotlib
- scikit-learn, hazm, gensim
- sentence-transformers
- arabic-reshaper, python-bidi
- plotly, scipy
