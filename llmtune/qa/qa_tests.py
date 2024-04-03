from typing import List, Union

import nltk
import numpy as np
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from transformers import DistilBertModel, DistilBertTokenizer

from llmtune.qa.generics import LLMQaTest, TestRegistry


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


@TestRegistry.register("summary_length")
class LengthTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "summary_length"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        return abs(len(ground_truth) - len(model_prediction))


@TestRegistry.register("jaccard_similarity")
class JaccardSimilarityTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "jaccard_similarity"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        set_ground_truth = set(ground_truth.lower())
        set_model_prediction = set(model_prediction.lower())

        intersection_size = len(set_ground_truth.intersection(set_model_prediction))
        union_size = len(set_ground_truth.union(set_model_prediction))

        similarity = intersection_size / union_size if union_size != 0 else 0
        return similarity


@TestRegistry.register("dot_product")
class DotProductSimilarityTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "dot_product"

    def _encode_sentence(self, sentence):
        tokens = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        embedding_ground_truth = self._encode_sentence(ground_truth)
        embedding_model_prediction = self._encode_sentence(model_prediction)
        dot_product_similarity = np.dot(embedding_ground_truth, embedding_model_prediction)
        return dot_product_similarity


@TestRegistry.register("rouge_score")
class RougeScoreTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "rouge_score"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(model_prediction, ground_truth)
        return float(scores["rouge1"].precision)


@TestRegistry.register("word_overlap")
class WordOverlapTest(LLMQaTest):
    @property
    def test_name(self) -> str:
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
        return overlap_percentage


class PosCompositionTest(LLMQaTest):
    def _get_pos_percent(self, text: str, pos_tags: List[str]) -> float:
        words = word_tokenize(text)
        tags = pos_tag(words)
        pos_words = [word for word, tag in tags if tag in pos_tags]
        total_words = len(text.split(" "))
        return round(len(pos_words) / total_words, 2)


@TestRegistry.register("verb_percent")
class VerbPercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "verb_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])


@TestRegistry.register("adjective_percent")
class AdjectivePercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "adjective_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["JJ", "JJR", "JJS"])


@TestRegistry.register("noun_percent")
class NounPercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "noun_percent"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ["NN", "NNS", "NNP", "NNPS"])


# Instantiate tests
# length_test = LengthTest()
# jaccard_similarity_test = JaccardSimilarityTest()
# dot_product_similarity_test = DotProductSimilarityTest()
# rouge_score_test = RougeScoreTest()
# word_overlap_test = WordOverlapTest()
# verb_percent_test = VerbPercent()
# adjective_percent_test = AdjectivePercent()
# noun_percent_test = NounPercent()
