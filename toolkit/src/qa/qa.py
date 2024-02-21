from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict



class LLMQaTest(ABC):
    @property
    @abstractmethod
    def test_name(self) -> str:
        pass

    @abstractmethod
    def get_metric(self, prompt:str, grount_truth:str, model_pred: str) -> Union[float, int, bool]:
        pass


class LLMTestSuite():
    def __init__(self, 
                 tests:List[LLMQaTest],
                 prompts:List[str],
                 ground_truths:List[str],
                 model_preds:List[str]) -> None:

        self.tests = tests
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.model_preds = model_preds

        self.test_results = {}

    def run_tests(self) -> Dict[str, List[Union[float, int, bool]]]:
        test_results = {}
        for test in zip(self.tests):
            metrics = []
            for prompt, ground_truth, model_pred in zip(self.prompts, self.ground_truths, self.model_preds):
                metrics.append(test.get_metric(prompt, ground_truth, model_pred))
            test_results[test.test_name] = metrics
        
        self.test_results = test_results

        return test_results
    
    @property
    def test_results(self):
        return self.test_results if self.test_results else self.run_tests() 
    

    def print_test_results(self):
        # TODO: format these!
        print(self.test_results)

    def save_test_results(self, path:str):
        # TODO: save these!
        pass