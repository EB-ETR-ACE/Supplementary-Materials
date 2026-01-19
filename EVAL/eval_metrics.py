#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py — Compute evaluation metrics for German "Leichte Sprache" (simplified language).

Metrics calculated:
- Readability: Flesch Reading Ease (Amstad), Wiener Sachtextformel.
- Complexity: Zipf frequency, syllable counts.
- Rule Violations: Passive voice, genitive case, subjunctive mood, abbreviations, long words/sentences.
- Semantic Similarity: BERTScore (if references provided).

Usage:
    python eval_metrics.py --input data.csv --output metrics.csv
    python eval_metrics.py --input data.jsonl --output metrics.csv --spacy-model de_core_news_lg
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
import pyphen
from wordfreq import zipf_frequency

# Optional BERTScore import
try:
    from bert_score import score as bertscore_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert_score not installed. BERTScore metrics will be skipped.", file=sys.stderr)

# --- Constants & Regex ---
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+", re.UNICODE)

@dataclass
class Metrics:
    """Container for all computed metrics per text sample."""
    # Readability Scores
    flesch_reading_ease: float
    wiener_sachtextformel: float
    avg_zipf_frequency: float

    # Rule Violation Counts
    count_abbreviations: int
    count_long_words_syllables_gt3: int
    count_long_sentences_tokens_gt10: int
    count_long_compounds: int
    count_complex_noun_phrases: int
    count_genitive_des_det: int
    count_passive_sentences: int
    count_subjunctive_tokens: int

    # Basic Statistics
    num_sentences: int
    num_words: int
    num_syllables: int

    # Semantic Similarity (Optional)
    bert_precision: Optional[float] = None
    bert_recall: Optional[float] = None
    bert_f1: Optional[float] = None

class Evaluator:
    def __init__(self, model_name: str = "de_core_news_lg"):
        print(f"Loading spaCy model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Error: spaCy model '{model_name}' not found. Please run: python -m spacy download {model_name}")
            sys.exit(1)
            
        self.pyphen_dic = pyphen.Pyphen(lang="de_DE")

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using Pyphen."""
        inserted = self.pyphen_dic.inserted(word)
        if not inserted:
            return 1 if WORD_RE.fullmatch(word) else 0
        return inserted.count("-") + 1

    def _percent(self, num: int, total: int) -> float:
        return (num / total) * 100.0 if total > 0 else 0.0

    def compute(self, text: str) -> Metrics:
        """Compute all metrics for a given text."""
        doc = self.nlp(text)
        
        # --- Basic Counts ---
        sentences = list(doc.sents)
        words_non_punct = [t for t in doc if not t.is_punct and not t.is_space]
        words_alpha = [t for t in words_non_punct if t.is_alpha]
        
        num_sentences = len(sentences)
        num_words = len(words_non_punct) # including numbers, symbols if not punct/space
        num_words_alpha = len(words_alpha)
        
        # --- Syllable & Word Length Analysis ---
        total_syllables = 0
        long_words_gt3_syll = 0
        polysyllabic_count = 0  # >= 3 syllables
        monosyllabic_count = 0  # == 1 syllable
        long_words_gt6_chars = 0 # >= 6 chars

        for token in words_non_punct:
            if len(token.text) >= 6:
                long_words_gt6_chars += 1
            
            if token.is_alpha:
                s_count = self._count_syllables(token.text)
                total_syllables += s_count
                
                if s_count == 1: monosyllabic_count += 1
                if s_count >= 3: polysyllabic_count += 1
                if s_count > 3:  long_words_gt3_syll += 1

        # --- Readability Formulas ---
        flesch = 0.0
        wiener = 0.0

        if num_sentences > 0 and num_words > 0:
            asl = num_words / num_sentences
            asw = total_syllables / num_words
            
            # Flesch Reading Ease (Amstad adaptation)
            flesch = 180.0 - asl - (58.5 * asw)

            # Wiener Sachtextformel (Variant 1)
            ms = self._percent(polysyllabic_count, num_words)
            iw = self._percent(long_words_gt6_chars, num_words)
            es = self._percent(monosyllabic_count, num_words)
            
            wiener = (0.1935 * ms) + (0.1672 * asl) + (0.1297 * iw) - (0.0327 * es) - 0.875

        # --- Zipf Frequency ---
        zipf_vals = [zipf_frequency(t.lemma_.lower(), "de") for t in words_alpha]
        avg_zipf = sum(zipf_vals) / len(zipf_vals) if zipf_vals else 0.0

        # --- Rule Violations ---
        # 1. Long sentences (>10 words)
        long_sent_count = sum(1 for s in sentences if len(s) > 10)
        
        # 2. Complex Noun Phrases (Noun with >2 children)
        complex_np_count = 0
        for t in doc:
            if t.pos_ == "NOUN" and len(list(t.children)) > 2:
                complex_np_count += 1
                
        # 3. Long Compounds (Noun > 15 chars)
        long_compound_count = sum(1 for t in doc if t.pos_ == "NOUN" and len(t.text) > 15)
        
        # 4. Genitive 'des'
        genitive_des_count = sum(1 for t in doc if t.lower_ == "des" and t.pos_ == "DET")
        
        # 5. Passive Voice (auxpass dependency)
        passive_sent_count = 0
        for sent in sentences:
            if any(t.dep_ == "auxpass" for t in sent):
                passive_sent_count += 1
                
        # 6. Subjunctive Mood
        subjunctive_count = 0
        for t in doc:
            moods = t.morph.get("Mood")
            if any(m.startswith("Sub") for m in moods):
                subjunctive_count += 1
                
        # 7. Abbreviations (Pattern matching dot inside word or ending dot)
        abbr_count = 0
        for t in doc:
            if re.match(r".*\..*\.", t.text):
                abbr_count += 1

        return Metrics(
            flesch_reading_ease=round(flesch, 2),
            wiener_sachtextformel=round(wiener, 2),
            avg_zipf_frequency=round(avg_zipf, 2),
            count_abbreviations=abbr_count,
            count_long_words_syllables_gt3=long_words_gt3_syll,
            count_long_sentences_tokens_gt10=long_sent_count,
            count_long_compounds=long_compound_count,
            count_complex_noun_phrases=complex_np_count,
            count_genitive_des_det=genitive_des_count,
            count_passive_sentences=passive_sent_count,
            count_subjunctive_tokens=subjunctive_count,
            num_sentences=num_sentences,
            num_words=num_words,
            num_syllables=total_syllables
        )

def compute_bertscore(candidates: List[str], references: List[str]) -> List[Tuple[float, float, float]]:
    """Computes BERTScore (Precision, Recall, F1) for a batch."""
    if not HAS_BERTSCORE or not candidates or not references:
        return [(None, None, None)] * len(candidates)
    
    # Filter out None references safely
    valid_indices = [i for i, r in enumerate(references) if r]
    if not valid_indices:
        return [(None, None, None)] * len(candidates)
        
    c_list = [candidates[i] for i in valid_indices]
    r_list = [references[i] for i in valid_indices]

    try:
        P, R, F1 = bertscore_score(c_list, r_list, lang="de", model_type="xlm-roberta-large", verbose=False)
        P, R, F1 = P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        print(f"Error computing BERTScore: {e}", file=sys.stderr)
        return [(None, None, None)] * len(candidates)

    results = [(None, None, None)] * len(candidates)
    for i, idx in enumerate(valid_indices):
        results[idx] = (float(P[i]), float(R[i]), float(F1[i]))
    return results

def load_data(path: str) -> Tuple[List[str], List[Optional[str]]]:
    """Loads text and reference data from CSV or JSONL."""
    texts = []
    refs = []
    
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError("CSV must contain 'text' column.")
        texts = df["text"].fillna("").astype(str).tolist()
        if "reference" in df.columns:
            refs = df["reference"].replace({pd.NA: None}).tolist()
        else:
            refs = [None] * len(texts)
            
    elif path.endswith(".jsonl") or path.endswith(".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
                refs.append(obj.get("reference"))
    else:
        # Plain text file
        with open(path, "r", encoding="utf-8") as f:
            texts = [f.read().strip()]
            refs = [None]
            
    return texts, refs

def main():
    parser = argparse.ArgumentParser(description="Calculate Leichte Sprache Metrics.")
    parser.add_argument("--input", required=True, help="Input file (.csv, .jsonl, .txt)")
    parser.add_argument("--output", required=True, help="Output CSV file for metrics")
    parser.add_argument("--summary", help="Optional JSON file for aggregated summary")
    parser.add_argument("--spacy-model", default="de_core_news_lg", help="SpaCy model to use")
    args = parser.parse_args()

    # Load Data
    print(f"Reading input: {args.input}")
    texts, refs = load_data(args.input)
    
    if not texts:
        print("No text found in input.", file=sys.stderr)
        sys.exit(0)

    # Initialize Evaluator
    evaluator = Evaluator(args.spacy_model)
    
    # Compute Linguistic Metrics
    results = []
    print("Computing linguistic metrics...")
    for txt in tqdm(texts):
        if not txt:
            # Empty text results in zero-metrics
            results.append(Metrics(0,0,0,0,0,0,0,0,0,0,0,0,0,0))
        else:
            results.append(evaluator.compute(txt))

    # Compute BERTScore (if references exist)
    if any(r for r in refs):
        print("Computing BERTScore...")
        bert_scores = compute_bertscore(texts, refs)
        for res, (p, r, f1) in zip(results, bert_scores):
            res.bert_precision = p
            res.bert_recall = r
            res.bert_f1 = f1

    # Create DataFrame
    data_list = [asdict(r) for r in results]
    df = pd.DataFrame(data_list)
    
    # Add Filename/ID columns for compatibility
    df.insert(0, "filename", [f"doc_{i}" for i in range(len(df))])

    # Save to CSV
    print(f"Saving per-document metrics to {args.output}")
    df.to_csv(args.output, index=False)

    # Save Summary
    if args.summary:
        summary_data = df.mean(numeric_only=True).to_dict()
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to {args.summary}")

if __name__ == "__main__":
    main()
