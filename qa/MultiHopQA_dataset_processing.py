from datasets import load_dataset
import openai
from sklearn.cluster import KMeans
import random
from typing import List, Dict, Any
import numpy as np
import json
from tqdm import tqdm

class HotpotQAProcessor:
    """Processor for HotpotQA train split (distractor), using only the first 10k examples."""

    def __init__(
        self,
        k: int = 10,
        samples_per_cluster: int = 10,
        embedding_model: str = "text-embedding-ada-002",
    ):
        """
        Parameters
        ----------
        k : int
            Number of clusters for KMeans.
        samples_per_cluster : int
            Samples per cluster per difficulty level.
        embedding_model : str
            OpenAI embedding model name.
        """
        self.k = k
        self.n = samples_per_cluster
        self.embedding_model = embedding_model
        self.dataset = None
        self.train_embeddings = None
        self.kmeans = None
        self.train_clusters = None

    def load_data(self):
        """Load the HotpotQA 'distractor' train split."""
        self.dataset = load_dataset(
            "hotpot_qa", "distractor", trust_remote_code=True
        )

    def embed_questions(self):
        """
        Embed the first up-to-10k train questions with openai>=1.20 embeddings API.
        """
        def batch_embed(questions: List[str]) -> np.ndarray:
            embs = []
            for q in tqdm(questions):
                resp = openai.embeddings.create(
                    model=self.embedding_model,
                    input=q
                )
                embs.append(resp.data[0].embedding)
            return np.array(embs)

        limit = min(50000, len(self.dataset["train"]))
        subset = self.dataset["train"].select(list(range(limit)))
        train_qs = [ex["question"] for ex in subset]
        self.train_embeddings = batch_embed(train_qs)

    def cluster_questions(self):
        """Cluster the embeddings into K clusters and store labels."""
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.kmeans.fit(self.train_embeddings)
        self.train_clusters = self.kmeans.labels_


    def random_sample_indices(self, per_level: int = 100) -> Dict[str, List[int]]:
        """
        Randomly sample up to `per_level` examples for each difficulty level
        (easy/medium/hard) from the *same* first-10k training subset.

        Returns
        -------
        Dict[str, List[int]]
            A mapping from difficulty level to the list of sampled indices
            (relative to the 10k‐subset).
        """
        limit = min(50000, len(self.dataset["train"]))
        train_subset = self.dataset["train"].select(list(range(limit)))

        difficulties = ["easy", "medium", "hard"]
        sampled_indices: Dict[str, List[int]] = {}

        for level in difficulties:
            # find all indices with this level
            candidates = [
                i for i, ex in enumerate(train_subset)
                if ex["level"] == level
            ]
            # sample up to per_level
            sampled_indices[level] = random.sample(
                candidates,
                k=min(per_level, len(candidates))
            )

        return sampled_indices

    def sample_by_difficulty(self) -> List[Dict[str, Any]]:
        """
        From the same first-10k subset, sample `n` examples per cluster per level.
        Returns a list of dicts with keys: id, question, answer, type, level,
        supporting_facts, context (flattened).
        """
        limit = min(50000, len(self.dataset["train"]))
        train_subset = self.dataset["train"].select(list(range(limit)))

        difficulties = ["easy", "medium", "hard"]
        sampled_indices: List[int] = []

        for level in difficulties:
            for cluster_id in range(self.k):
                candidates = [
                    i for i, ex in enumerate(train_subset)
                    if self.train_clusters[i] == cluster_id and ex["level"] == level
                ]
                chosen = random.sample(candidates, k=min(self.n, len(candidates)))
                sampled_indices.extend(chosen)

        samples: List[Dict[str, Any]] = []
        for idx in sampled_indices:
            ex = train_subset[idx]
            # Take only the first sentence from each doc for context
            titles = ex["context"]["title"]
            sentences = ex["context"]["sentences"]
            ctx_lines = [
                f"{title}: {sents[0].strip()}"
                for title, sents in zip(titles, sentences) if sents
            ]
            context_str = "\n".join(ctx_lines)

            samples.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "supporting_facts": ex["supporting_facts"],
                "context": context_str,
            })
        return samples

    def sample_by_difficulty_light(
        self,
        per_level: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Randomly sample up to `per_level` examples for each difficulty level
        (easy / medium / hard) from the first 10k examples, and return the
        fully–built sample dicts (id, question, answer, type, level,
        supporting_facts, context).
        """
        # 1) restrict to first 10k
        limit = min(10000, len(self.dataset["train"]))
        subset = self.dataset["train"].select(list(range(limit)))

        difficulties = ["easy", "medium", "hard"]
        samples: List[Dict[str, Any]] = []

        for level in difficulties:
            # collect all indices with this level
            candidates = [
                i for i, ex in enumerate(subset)
                if ex["level"] == level
            ]
            # sample up to per_level
            chosen = random.sample(candidates, k=min(per_level, len(candidates)))

            # build sample dicts immediately
            for idx in chosen:
                ex = subset[idx]
                titles = ex["context"]["title"]
                sents = ex["context"]["sentences"]
                ctx_lines = [
                    f"{title}: {s[0].strip()}"
                    for title, s in zip(titles, sents) if s
                ]
                context_str = "\n".join(ctx_lines)

                samples.append({
                    "id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "type": ex["type"],
                    "level": ex["level"],
                    "supporting_facts": ex["supporting_facts"],
                    "context": context_str,
                })

        return samples



    def save_to_json(self, samples: List[Dict[str, Any]], json_path: str) -> None:
        """Dump the samples list to `json_path` as UTF-8 JSON."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example end-to-end run
    processor = HotpotQAProcessor(k=10, samples_per_cluster=10)
    print("Loading data…")
    processor.load_data()
    print("Sampling data…")
    light_samples = processor.sample_by_difficulty_light(per_level=100)
    print(len(light_samples))  # ≤ 300
    print("Save data…")
    processor.save_to_json(light_samples, "light_hotpotqa.json")
