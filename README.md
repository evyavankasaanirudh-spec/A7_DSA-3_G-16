# Intelligent Multi-Pattern Search and Document Analytics Engine

## Project Overview

The Intelligent Multi-Pattern Search and Document Analytics Engine is a Python-based search system developed using data structures and string-matching algorithms.

The system processes a collection of text documents and supports efficient keyword search, multi-pattern matching, document validation, and relevance-based ranking.

The project uses the 20 Newsgroups dataset for experimentation and evaluation.

## Objectives

- Load and process a large collection of text documents.
- Convert raw text into searchable tokens.
- Build an inverted index for fast document retrieval.
- Implement single-word and multi-word search.
- Implement the Aho–Corasick multi-pattern matching algorithm.
- Validate search results using token-based matching.
- Rank documents using term frequency.
- Compare the performance of different search techniques.

## Dataset

### Dataset Name

20 Newsgroups Dataset

### Dataset Description

The dataset contains text documents collected from different newsgroup categories.

### Dataset Details

- Total documents: 11,314
- Number of categories: 20
- Data type: Text documents
- Source: Scikit-learn dataset collection

The dataset was loaded using:

```python
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)
