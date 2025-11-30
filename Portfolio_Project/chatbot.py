import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

''' A CPW Regulation Retrieval and QA System 
This script implements a retrieval-based QA system over Colorado Parks and Wildlife regulations.
It parses a text file of regulations into sections, builds an embedding index for retrieval,
and uses a fine-tuned DistilBERT model to answer user questions based on retrieved sections.'''


# Configuration
CPW_TEXT_PATH = "CPW_regulations.txt"
SECTIONS_JSON_PATH = "cpw_sections.json"
EMBEDDINGS_NPY_PATH = "cpw_section_embeddings.npy"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_PATH = "./model_output"  # output from training

TOP_K_SECTIONS = 3  # how many sections to retrieve per query


# Parse CPW text into sections
# When I created the CPW_regulations.txt file, I used this BEGIN/END SECTION format:
# The regex pattern simply looks for these markers to split the text.

SECTION_BEGIN_RE = re.compile(r"^\s*BEGIN SECTION -- (.+?)\s*$")
SECTION_END_RE = re.compile(r"^\s*END SECTION -- (.+?)\s*$")


def parse_cpw_sections(text_path):
    sections = []
    current_title = None
    current_lines = []

    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            begin_match = SECTION_BEGIN_RE.match(line)
            end_match = SECTION_END_RE.match(line)

            if begin_match:
                # If we were already inside a section, close it
                if current_title is not None and current_lines:
                    sections.append(
                        {
                            "id": len(sections),
                            "title": current_title.strip(),
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_title = begin_match.group(1).strip()
                current_lines = []
            elif end_match:
                # End current section
                if current_title is not None:
                    sections.append(
                        {
                            "id": len(sections),
                            "title": current_title.strip(),
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_title = None
                current_lines = []
            else:
                # Normal line: if we're inside a section, collect it
                if current_title is not None:
                    current_lines.append(line.rstrip("\n"))

    # Safety: if file ended without an END SECTION
    if current_title is not None and current_lines:
        sections.append(
            {
                "id": len(sections),
                "title": current_title.strip(),
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections


# Build or load the index
def build_or_load_index():
    # Try to load cached sections + embeddings first
    try:
        with open(SECTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            sections = json.load(f)
        embeddings = np.load(EMBEDDINGS_NPY_PATH)
        print(f"Loaded {len(sections)} sections and embeddings from disk.")
        return sections, embeddings
    except FileNotFoundError:
        print("No saved index found. Parsing text and building embeddings...")

    # Parse sections from the raw text file
    sections = parse_cpw_sections(CPW_TEXT_PATH)
    print(f"Parsed {len(sections)} sections from CPW text.")

    # Build embeddings
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [s["text"] for s in sections]
    print("Encoding section embeddings...")
    embeddings = embed_model.encode(texts, normalize_embeddings=True)

    # Save for reuse
    with open(SECTIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_NPY_PATH, embeddings)

    print("Index built and saved to disk.")
    return sections, embeddings


# Retrieve relevant sections
def retrieve_sections(question, sections, embeddings, top_k=TOP_K_SECTIONS):
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Encode the question
    q_emb = embed_model.encode([question], normalize_embeddings=True)[0]

    # Cosine similarity with pre-normalized embeddings = dot product
    scores = embeddings @ q_emb
    top_indices = np.argsort(-scores)[:top_k]

    retrieved = []
    for idx in top_indices:
        sec = sections[idx]
        retrieved.append(
            {
                "id": sec["id"],
                "title": sec["title"],
                "text": sec["text"],
                "score": float(scores[idx]),
            }
        )
    return retrieved


# Build QA pipeline
def build_qa_pipeline():
    qa_pipe = pipeline(
        "question-answering",
        model=QA_MODEL_PATH,
        tokenizer=QA_MODEL_PATH,
    )
    return qa_pipe

# Answer the question using retrieved sections
def answer_question(question, sections, embeddings, qa_pipe, top_k=TOP_K_SECTIONS):
    retrieved = retrieve_sections(question, sections, embeddings, top_k=top_k)

    best_answer = None
    best_score = -1.0

    for r in retrieved:
        result = qa_pipe(
            question=question,
            context=r["text"],
        )
        score = float(result["score"])
        answer_text = result["answer"]

        if score > best_score:
            best_score = score
            best_answer = {
                "answer": answer_text,
                "score": score,
                "section_id": r["id"],
                "section_title": r["title"],
                "section_score": r["score"],
            }

    return {
        "question": question,
        "best_answer": best_answer,
        "retrieved_sections": retrieved,
    }


# Main interactive loop is a CLI chatbot
def main():
    print("---CSC 525 Portfolio Project ---")
    print("This is a CPW Regulation Retrieval and QA Chatbot System.")
    print("For this to work training.py must have been run first which will save a model in ./model_output")
    print("Loading / building CPW retrieval index...")
    sections, embeddings = build_or_load_index()

    print("Loading QA pipeline from fine-tuned model...")
    qa_pipe = build_qa_pipeline()

    print("\nCPW Retrieval QA Ready! Type a question, or 'quit' to exit.\n")

    while True:
        q = input("Question: ").strip()
        if not q or q.lower() in {"q", "quit", "exit"}:
            break

        result = answer_question(q, sections, embeddings, qa_pipe, top_k=TOP_K_SECTIONS)
        best = result["best_answer"]

        if best is None or best["answer"].strip() == "":
            print("\nI couldn't find a specific answer in the brochure.\n")
            continue

        print(f"\nAnswer: {best['answer']} (score: {best['score']:.3f})")
        print(f"   From section: {best['section_title']} (retrieval score: {best['section_score']:.3f})\n")


if __name__ == "__main__":
    main()