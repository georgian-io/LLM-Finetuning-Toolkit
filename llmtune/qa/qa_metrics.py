from abc import ABC, abstractmethod
from typing import List, Union
import nltk
import numpy as np
import torch
from langchain.evaluation import JsonValidityEvaluator
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from transformers import DistilBertModel, DistilBertTokenizer


json_validity_evaluator = JsonValidityEvaluator()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


class LLMQaMetric(ABC):
    """
    Abstract base class for a metric. A metric can be computed over a single
    data instance, and outputs a scalar value (integer or float).
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(self, prompt: str, grount_truth: str, model_pred: str) -> Union[float, int]:
        pass


class QaMetricRegistry:
    """Provides a registry that maps metric names to metric classes.
    A user can provide a list of metrics by name, and the registry will convert
    that into a list of metric objects.
    """
    registry = {}

    @classmethod
    def register(cls, *names):
        def inner_wrapper(wrapped_class):
            for name in names:
                cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_metrics_from_list(cls, metric_names: List[str]) -> List[LLMQaMetric]:
        return [cls.registry[metric]() for metric in metric_names]


@QaMetricRegistry.register("summary_length")
class LengthMetric(LLMQaMetric):
    @property
    def metric_name(self) -> str:
        return "summary_length"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        return abs(len(ground_truth) - len(model_prediction))


@QaMetricRegistry.register("jaccard_similarity")
class JaccardSimilarityMetric(LLMQaMetric):
    @property
    def metric_name(self) -> str:
        return "jaccard_similarity"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        set_ground_truth = set(ground_truth.lower())
        set_model_prediction = set(model_prediction.lower())

        intersection_size = len(set_ground_truth.intersection(set_model_prediction))
        union_size = len(set_ground_truth.union(set_model_prediction))

        similarity = intersection_size / union_size if union_size != 0 else 0
        return float(similarity)


@QaMetricRegistry.register("dot_product")
class DotProductSimilarityMetric(LLMQaMetric):
    """Encodes both the ground truth and model prediction using DistilBERT, and
    computes the dot product similarity between the two embeddings."""

    def __init__(self):
        model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

    @property
    def metric_name(self) -> str:
        return "dot_product"

    def _encode_sentence(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        embedding_ground_truth = self._encode_sentence(ground_truth)
        embedding_model_prediction = self._encode_sentence(model_prediction)
        dot_product_similarity = np.dot(embedding_ground_truth, embedding_model_prediction)
        return float(dot_product_similarity)


@QaMetricRegistry.register("rouge_score")
class RougeScoreMetric(LLMQaMetric):
    @property
    def metric_name(self) -> str:
        return "rouge_score"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(model_prediction, ground_truth)
        return float(scores["rouge1"].precision)


@QaMetricRegistry.register("word_overlap")
class WordOverlapMetric(LLMQaMetric):
    @property
    def metric_name(self) -> str:
        return "word_overlap"

    def _remove_stopwords(self, text: str) -> str:
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return " ".join(filtered_text)

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        cleaned_model_prediction = self._remove_stopwords(model_prediction)
        cleaned_ground_truth = self._remove_stopwords(ground_truth)

        words_model_prediction = set(cleaned_model_prediction.split())
        words_ground_truth = set(cleaned_ground_truth.split())

        common_words = words_model_prediction.intersection(words_ground_truth)
        overlap_percentage = (len(common_words) / len(words_ground_truth)) * 100
        return float(overlap_percentage)


@QaMetricRegistry.register("json_valid")
class JSONValidityMetric(LLMQaMetric):
    """
    Checks to see if valid json can be parsed from the model output, according
    to langchain_core.utils.json.parse_json_markdown
    The JSON can be wrapped in markdown and this test will still pass
    """

    @property
    def metric_name(self) -> str:
        return "json_valid"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        result = json_validity_evaluator.evaluate_strings(prediction=model_prediction)
        binary_res = result["score"]
        return float(binary_res)


class PosCompositionMetric(LLMQaMetric):
    def _get_pos_percent(self, text: str, pos_tags: List[str]) -> float:
        words = word_tokenize(text)
        tags = pos_tag(words)
        pos_words = [word for word, tag in tags if tag in pos_tags]
        total_words = len(text.split(" "))
        return round(len(pos_words) / total_words, 2)


@QaMetricRegistry.register("verb_percent")
class VerbPercentMetric(PosCompositionMetric):
    @property
    def metric_name(self) -> str:
        return "verb_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])


@QaMetricRegistry.register("adjective_percent")
class AdjectivePercentMetric(PosCompositionMetric):
    @property
    def metric_name(self) -> str:
        return "adjective_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["JJ", "JJR", "JJS"])


@QaMetricRegistry.register("noun_percent")
class NounPercentMetric(PosCompositionMetric):
    @property
    def metric_name(self) -> str:
        return "noun_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["NN", "NNS", "NNP", "NNPS"])


# Instantiate tests
# length_test = LengthMetric()
# jaccard_similarity_test = JaccardSimilarityMetric()
# dot_product_similarity_test = DotProductSimilarityMetric()
# rouge_score_test = RougeScoreMetric()
# word_overlap_test = WordOverlapMetric()
# verb_percent_test = VerbPercentMetric()
# adjective_percent_test = AdjectivePercentMetric()
# noun_percent_test = NounPercentMetric()
