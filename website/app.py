from flask import Flask, render_template, request
from collections import defaultdict, Counter
import re
import time
from sklearn.datasets import fetch_20newsgroups

app = Flask(__name__)


# ============================================================
# DATASET
# ============================================================

dataset = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

documents = dataset.data
labels = dataset.target
target_names = dataset.target_names


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b[a-zA-Z0-9]+\b", text)


# ============================================================
# INVERTED INDEX
# ============================================================

def build_inverted_index(documents):
    inverted_index = defaultdict(set)

    for doc_id, document in enumerate(documents):
        tokens = tokenize(document)

        for token in set(tokens):
            inverted_index[token].add(doc_id)

    return dict(inverted_index)


print("Building inverted index...")
index_start = time.perf_counter()

inverted_index = build_inverted_index(documents)

index_build_time = time.perf_counter() - index_start

print(f"Inverted index built in {index_build_time:.6f} seconds")
print(f"Documents: {len(documents)}")
print(f"Unique terms: {len(inverted_index)}")


# ============================================================
# INVERTED INDEX SEARCH
# ============================================================

def search_documents(query, inverted_index, mode="AND"):

    query_terms = tokenize(query)

    if not query_terms:
        return set()

    query_terms = list(dict.fromkeys(query_terms))

    # Remove terms that don't exist in the index
    existing_terms = [
        term for term in query_terms
        if term in inverted_index
    ]

    if mode == "AND":

        # Every query term must be present
        if len(existing_terms) != len(query_terms):
            return set()

        result = inverted_index[existing_terms[0]].copy()

        for term in existing_terms[1:]:
            result &= inverted_index[term]

        return result

    else:

        # OR → at least one query term must be present
        result = set()

        for term in existing_terms:
            result |= inverted_index[term]

        return result


# ============================================================
# AHO-CORASICK
# ============================================================

class AhoCorasick:

    def __init__(self):
        self.goto = [{}]
        self.fail = [0]
        self.output = [[]]

    def add_pattern(self, pattern):

        node = 0

        for char in pattern:

            if char not in self.goto[node]:

                self.goto[node][char] = len(self.goto)

                self.goto.append({})
                self.fail.append(0)
                self.output.append([])

            node = self.goto[node][char]

        self.output[node].append(pattern)

    def build_failure_links(self):

        queue = []

        for char, next_node in self.goto[0].items():

            self.fail[next_node] = 0
            queue.append(next_node)

        while queue:

            current = queue.pop(0)

            for char, next_node in self.goto[current].items():

                queue.append(next_node)

                failure_node = self.fail[current]

                while (
                    failure_node != 0
                    and char not in self.goto[failure_node]
                ):
                    failure_node = self.fail[failure_node]

                if char in self.goto[failure_node]:
                    self.fail[next_node] = self.goto[failure_node][char]
                else:
                    self.fail[next_node] = 0

                self.output[next_node].extend(
                    self.output[self.fail[next_node]]
                )

    def search(self, text):

        text = text.lower()

        node = 0
        found_patterns = []

        for char in text:

            while node != 0 and char not in self.goto[node]:
                node = self.fail[node]

            if char in self.goto[node]:
                node = self.goto[node][char]
            else:
                node = 0

            if self.output[node]:
                found_patterns.extend(self.output[node])

        return found_patterns


# ============================================================
# AHO-CORASICK TOKEN-BASED SEARCH
# ============================================================

def search_multiple_patterns_token_based(documents, patterns):

    patterns = [pattern.lower() for pattern in patterns]

    ac = AhoCorasick()

    for pattern in patterns:
        ac.add_pattern(pattern)

    ac.build_failure_links()

    results = defaultdict(set)

    for doc_id, document in enumerate(documents):

        tokens = tokenize(document)
        normalized_text = " ".join(tokens)

        matches = ac.search(normalized_text)

        matched_patterns = set()
        token_set = set(tokens)

        for pattern in matches:

            if pattern in token_set:
                matched_patterns.add(pattern)

        if matched_patterns:
            results[doc_id] = matched_patterns

    return dict(results)


def aho_and_search(documents, patterns):

    patterns = [pattern.lower() for pattern in patterns]

    results = search_multiple_patterns_token_based(
        documents,
        patterns
    )

    required_patterns = set(patterns)

    matching_documents = set()

    for doc_id, matched_patterns in results.items():

        if required_patterns.issubset(matched_patterns):
            matching_documents.add(doc_id)

    return matching_documents


# ============================================================
# RANKED SEARCH
# ============================================================

def ranked_search(query, mode="AND"):

    query_terms = tokenize(query)

    matching_documents = search_documents(
        query,
        inverted_index,
        mode
    )

    ranked_results = []

    for doc_id in matching_documents:

        tokens = tokenize(documents[doc_id])
        frequencies = Counter(tokens)

        score = sum(
            frequencies.get(term, 0)
            for term in query_terms
        )

        ranked_results.append({
            "Document_ID": doc_id,
            "Score": score
        })

    ranked_results.sort(
        key=lambda x: x["Score"],
        reverse=True
    )

    return ranked_results

# ============================================================
# DOCUMENT ANALYTICS
# ============================================================

def get_document_preview(doc_id, query=""):

    text = documents[doc_id].strip()

    if not text:
        return "This document contains no visible text."

    # Keep preview manageable
    preview_length = 450

    if len(text) > preview_length:
        preview = text[:preview_length].rsplit(" ", 1)[0] + "..."
    else:
        preview = text

    return preview


def get_result_analytics(result_ids):

    if not result_ids:
        return {
            "top_category": "—",
            "top_category_count": 0,
            "average_length": 0,
            "longest_document": 0
        }

    categories = [
        target_names[labels[doc_id]]
        for doc_id in result_ids
    ]

    category_counts = Counter(categories)

    top_category, top_category_count = (
        category_counts.most_common(1)[0]
    )

    lengths = [
        len(documents[doc_id])
        for doc_id in result_ids
    ]

    return {
        "top_category": top_category,
        "top_category_count": top_category_count,
        "average_length": round(sum(lengths) / len(lengths), 2),
        "longest_document": max(lengths)
    }


def get_category_distribution(result_ids):

    categories = Counter(
        target_names[labels[doc_id]]
        for doc_id in result_ids
    )

    return categories.most_common()


# ============================================================
# PERFORMANCE DATA
# ============================================================

performance_data = {
    "Inverted Index Build": 1.589919,
    "Inverted Index Query": 0.000120,
    "Aho-Corasick Build": 0.000107,
    "Aho-Corasick Search": 1.688909,
    "Ranked Search": 0.060099
}

algorithm_comparison = {
    "Inverted Index": {
        "build": 1.589919,
        "search": 0.000120,
        "description": "Fast term-based retrieval using a pre-built inverted index."
    },
    "Aho-Corasick": {
        "build": 0.000107,
        "search": 1.688909,
        "description": "Multi-pattern matching using a finite-state automaton."
    },
    "Ranked Search": {
        "build": 0,
        "search": 0.060099,
        "description": "Ranks matching documents using term-frequency scores."
    }
}

# ============================================================
# ALGORITHM INFORMATION
# ============================================================

algorithm_info = {
    "Inverted Index": {
        "title": "Inverted Index",
        "description":
            "Uses a term-to-document mapping for fast "
            "multi-term retrieval using set intersection."
    },

    "Aho-Corasick": {
        "title": "Aho-Corasick",
        "description":
            "Uses a finite-state automaton to search "
            "multiple patterns efficiently."
    },

    "Ranked Search": {
        "title": "Ranked Search",
        "description":
            "Retrieves matching documents and ranks them "
            "using term-frequency scores."
    }
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []

    query = ""

    algorithm = "Inverted Index"

    search_mode = "AND"

    execution_time = 0

    result_count = 0

    analytics = {
        "top_category": "—",
        "top_category_count": 0,
        "average_length": 0,
        "longest_document": 0
    }

    category_distribution = []

    matching_category_count = 0

    top_category = ("N/A", 0)

    top_score = 0

    selected_document = None

    selected_document_id = None

    search_performance = None

    if request.method == "POST":

        query = request.form.get(
            "query",
            ""
        ).strip()

        algorithm = request.form.get(
            "algorithm",
            "Inverted Index"
        )

        search_mode = request.form.get(
            "search_mode",
            "AND"
        )

        start_time = time.perf_counter()

        matching_documents = set()

        ranked_results = []

        if query:

            patterns = tokenize(query)

            # ------------------------------------------------
            # INVERTED INDEX
            # ------------------------------------------------

            if algorithm == "Inverted Index":

                matching_documents = search_documents(
                    query,
                    inverted_index,
                    search_mode
                )

                sorted_ids = sorted(
                    matching_documents
                )

                for doc_id in sorted_ids[:20]:

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": "-",
                        "preview": get_document_preview(
                            doc_id,
                            query
                        )
                    })

            # ------------------------------------------------
            # AHO-CORASICK
            # ------------------------------------------------

            elif algorithm == "Aho-Corasick":

                matching_documents = aho_and_search(
                    documents,
                    patterns
                )

                sorted_ids = sorted(
                    matching_documents
                )

                for doc_id in sorted_ids[:20]:

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": "-",
                        "preview": get_document_preview(
                            doc_id,
                            query
                        )
                    })

            # ------------------------------------------------
            # RANKED SEARCH
            # ------------------------------------------------

            elif algorithm == "Ranked Search":

                ranked_results = ranked_search(
                    query,
                    search_mode
                )

                matching_documents = {
                    item["Document_ID"]
                    for item in ranked_results
                }

                for item in ranked_results[:20]:

                    doc_id = item["Document_ID"]

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": item["Score"],
                        "preview": get_document_preview(
                            doc_id,
                            query
                        )
                    })

                if ranked_results:
                    top_score = ranked_results[0]["Score"]

            result_count = len(
                matching_documents
            )

            analytics = get_result_analytics(
                matching_documents
            )

            category_distribution = get_category_distribution(
                matching_documents
            )

            top_category = (
                category_distribution[0]
                if category_distribution
                else ("N/A", 0)
            )

            matching_category_count = len(category_distribution)

            # -----------------------------------------------
            # TOP RESULT
            # -----------------------------------------------

            if results:

                selected_document_id = results[0]["id"]

                selected_document = {
                    "id": selected_document_id,
                    "category": target_names[
                        labels[selected_document_id]
                    ],
                    "text": documents[
                        selected_document_id
                    ]
                }

        execution_time = (
            time.perf_counter() - start_time
        )

        search_performance = round(
            execution_time * 1000,
            4
        )

    return render_template(

        "index.html",

        results=results,

        query=query,

        algorithm=algorithm,

        search_mode=search_mode,

        execution_time=execution_time,

        search_performance=search_performance,

        result_count=result_count,

        document_count=len(documents),

        term_count=len(inverted_index),

        category_count=len(target_names),

        matching_category_count=matching_category_count,

        performance=performance_data,
        algorithm_comparison=algorithm_comparison,

        analytics=analytics,

        category_distribution=category_distribution,

        top_category=top_category,

        top_score=top_score,

        selected_document=selected_document,

        algorithm_info=algorithm_info
    )

# ============================================================
# DOCUMENT API
# ============================================================

@app.route("/api/document/<int:doc_id>")
def get_document(doc_id):

    if doc_id < 0 or doc_id >= len(documents):
        return {
            "error": "Document not found"
        }, 404

    return {
        "document_id": doc_id,
        "category": target_names[labels[doc_id]],
        "length": len(documents[doc_id]),
        "text": documents[doc_id]
    }

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )