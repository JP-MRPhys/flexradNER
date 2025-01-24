from sentence_transformers import SentenceTransformer, util
import difflib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class EvaluationMetrics:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    ### Keyword Evaluation Metrics ###
    def compute_precision_at_k(self, true_keywords, predicted_keywords, k):
        precisions = []
        for true, pred in zip(true_keywords, predicted_keywords):
            pred_at_k = pred[:k]
            precision = len(set(true) & set(pred_at_k)) / k
            precisions.append(precision)
        return np.mean(precisions)

    def compute_ndcg(self, true_keywords, predicted_keywords, k):
        def dcg(rel):
            return sum((2 ** rel_i - 1) / np.log2(i + 2) for i, rel_i in enumerate(rel))

        ndcgs = []
        for true, pred in zip(true_keywords, predicted_keywords):
            relevance = [1 if p in true else 0 for p in pred[:k]]
            ideal_relevance = sorted(relevance, reverse=True)
            actual_dcg = dcg(relevance)
            ideal_dcg = dcg(ideal_relevance)
            ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def compute_map(self, true_keywords, predicted_keywords):
        ap_scores = []
        for true, pred in zip(true_keywords, predicted_keywords):
            hits = 0
            precisions = []
            for i, p in enumerate(pred, start=1):
                if p in true:
                    hits += 1
                    precisions.append(hits / i)
            if hits > 0:
                ap_scores.append(np.mean(precisions))
            else:
                ap_scores.append(0)
        return np.mean(ap_scores)

    ### Explanation Evaluation Metrics ###
    def compute_explanation_similarity(self, true_explanations, predicted_explanations):
        """
        Compute similarity between true and predicted explanations for each term using cosine similarity.
        """
        similarities = {}
        for term, true_explanation in true_explanations.items():
            if term in predicted_explanations:
                true_embedding = self.similarity_model.encode(true_explanation, convert_to_tensor=True)
                pred_embedding = self.similarity_model.encode(predicted_explanations[term], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(true_embedding, pred_embedding).item()
                similarities[term] = similarity
        return similarities

    def compute_difflib_similarity(self, true_explanations, predicted_explanations):
        """
        Compute similarity between true and predicted explanations for each term using difflib.
        """
        similarities = {}
        for term, true_explanation in true_explanations.items():
            if term in predicted_explanations:
                similarity = difflib.SequenceMatcher(None, true_explanation, predicted_explanations[term]).ratio()
                similarities[term] = similarity
        return similarities

    def evaluate_explanation_similarity(self, true_explanations, predicted_explanations):
        """
        Return average similarity for explanations across all terms using multiple methods.
        """
        cosine_similarities = self.compute_explanation_similarity(true_explanations, predicted_explanations)
        difflib_similarities = self.compute_difflib_similarity(true_explanations, predicted_explanations)

        average_cosine_similarity = (
            np.mean(list(cosine_similarities.values())) if cosine_similarities else 0
        )
        average_difflib_similarity = (
            np.mean(list(difflib_similarities.values())) if difflib_similarities else 0
        )

        return {
            "average_cosine_similarity": average_cosine_similarity,
            "average_difflib_similarity": average_difflib_similarity,
            "cosine_per_term_similarity": cosine_similarities,
            "difflib_per_term_similarity": difflib_similarities,
        }

    ### Precision, Recall, and F1 Score ###
    def compute_precision(self, true_keywords, predicted_keywords):
        """
        Compute precision for keyword extraction.
        """
        true_flat = [item for sublist in true_keywords for item in sublist]
        pred_flat = [item for sublist in predicted_keywords for item in sublist]
        return precision_score(true_flat, pred_flat, average='binary', zero_division=0)

    def compute_recall(self, true_keywords, predicted_keywords):
        """
        Compute recall for keyword extraction.
        """
        true_flat = [item for sublist in true_keywords for item in sublist]
        pred_flat = [item for sublist in predicted_keywords for item in sublist]
        return recall_score(true_flat, pred_flat, average='binary', zero_division=0)

    def compute_f1_score(self, true_keywords, predicted_keywords):
        """
        Compute F1 score for keyword extraction.
        """
        true_flat = [item for sublist in true_keywords for item in sublist]
        pred_flat = [item for sublist in predicted_keywords for item in sublist]
        return f1_score(true_flat, pred_flat, average='binary', zero_division=0)

