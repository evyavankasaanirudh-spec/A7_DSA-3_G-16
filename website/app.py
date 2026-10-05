from flask import Flask, render_template, request
from collections import defaultdict, Counter, deque
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


inverted_index = build_inverted_index(documents)


# ============================================================
# INVERTED INDEX SEARCH
# ============================================================

def search_documents(query, inverted_index):
    query_terms = tokenize(query)

    if not query_terms:
        return set()

    query_terms = list(dict.fromkeys(query_terms))

    for term in query_terms:
        if term not in inverted_index:
            return set()

    result = inverted_index[query_terms[0]].copy()

    for term in query_terms[1:]:
        result &= inverted_index[term]

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

        queue = deque()

        for char, next_node in self.goto[0].items():

            self.fail[next_node] = 0
            queue.append(next_node)

        while queue:

            current = queue.popleft()

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
# AHO-CORASICK SEARCH
# ============================================================

def aho_corasick_search(documents, patterns):

    patterns = [p.lower() for p in patterns]

    ac = AhoCorasick()

    for pattern in patterns:
        ac.add_pattern(pattern)

    ac.build_failure_links()

    results = defaultdict(set)

    for doc_id, document in enumerate(documents):

        matches = ac.search(document)

        for pattern in matches:
            results[doc_id].add(pattern)

    return dict(results)


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
# RANKING
# ============================================================

def ranked_search(query):

    query_terms = tokenize(query)

    matching_documents = search_documents(
        query,
        inverted_index
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
# PERFORMANCE DATA
# ============================================================

performance_data = {
    "Inverted Index Build": 1.589919,
    "Inverted Index Query": 0.000120,
    "Aho-Corasick Build": 0.000107,
    "Aho-Corasick Search": 1.688909,
    "Ranked Search": 0.060099
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    query = ""
    algorithm = "Inverted Index"
    execution_time = 0
    result_count = 0

    if request.method == "POST":

        query = request.form.get("query", "").strip()

        algorithm = request.form.get(
            "algorithm",
            "Inverted Index"
        )

        start_time = time.perf_counter()

        if query:

            patterns = tokenize(query)

            if algorithm == "Inverted Index":

                matching_documents = search_documents(
                    query,
                    inverted_index
                )

                for doc_id in sorted(matching_documents)[:20]:

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": "-"
                    })

            elif algorithm == "Aho-Corasick":

                matching_documents = aho_and_search(
                    documents,
                    patterns
                )

                for doc_id in sorted(matching_documents)[:20]:

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": "-"
                    })

            elif algorithm == "Ranked Search":

                ranked_results = ranked_search(query)

                for item in ranked_results[:20]:

                    doc_id = item["Document_ID"]

                    results.append({
                        "id": doc_id,
                        "category": target_names[labels[doc_id]],
                        "score": item["Score"]
                    })

            result_count = (
                len(matching_documents)
                if algorithm != "Ranked Search"
                else len(ranked_results)
            )

        execution_time = time.perf_counter() - start_time

    return render_template(
        "index.html",
        results=results,
        query=query,
        algorithm=algorithm,
        result_count=result_count,
        execution_time=execution_time,
        document_count=len(documents),
        term_count=len(inverted_index),
        category_count=len(target_names),
        performance=performance_data
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )