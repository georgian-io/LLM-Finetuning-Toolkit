from src.qa.qa import LLMQaTest
from typing import Union, List, Tuple, Dict
import torch 
from transformers import DistilBertModel, DistilBertTokenizer
import nltk 
import numpy as np 
from rouge_score import rouge_scorer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class LengthTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Summary Length Test"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        return abs(len(ground_truth) - len(model_prediction))


class JaccardSimilarityTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Jaccard Similarity"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        set_ground_truth = set(ground_truth.lower())
        set_model_prediction = set(model_prediction.lower())

        intersection_size = len(set_ground_truth.intersection(set_model_prediction))
        union_size = len(set_ground_truth.union(set_model_prediction))

        similarity = intersection_size / union_size if union_size != 0 else 0
        return similarity

class DotProductSimilarityTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Semantic Similarity"

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

class RougeScoreTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Rouge Score"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> Union[float, int, bool]:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(model_prediction, ground_truth)
        return float(scores['rouge1'].precision)

class WordOverlapTest(LLMQaTest):
    @property
    def test_name(self) -> str:
        return "Word Overlap Test"
    
    def _remove_stopwords(self, text: str) -> str:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)

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

class VerbPercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "Verb Composition"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

class AdjectivePercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "Adjective Composition"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ['JJ', "JJR", "JJS"])

class NounPercent(PosCompositionTest):
    @property
    def test_name(self) -> str:
        return "Noun Composition"

    def get_metric(self, prompt: str, ground_truth: str, model_prediction: str) -> float:
        return self._get_pos_percent(model_prediction, ['NN', 'NNS', 'NNP', 'NNPS'])

# Instantiate tests
# length_test = LengthTest()
# jaccard_similarity_test = JaccardSimilarityTest()
# dot_product_similarity_test = DotProductSimilarityTest()
# rouge_score_test = RougeScoreTest()
# word_overlap_test = WordOverlapTest()
# verb_percent_test = VerbPercent()
# adjective_percent_test = AdjectivePercent()
# noun_percent_test = NounPercent()
