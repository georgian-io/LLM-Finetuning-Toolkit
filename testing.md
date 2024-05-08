## LLM Testing Guidebook

Below, we outline an initial guide to testing LLMs.
LLMs are some of the hardest software systems to test, and have some unique challenges when compared to other ML systems.

## Motivation: Why should we test LLMs?

Companies can be held [liable for their chatbot's outputs](https://www.cbc.ca/news/canada/british-columbia/air-canada-chatbot-lawsuit-1.7116416).
Additionally, LLMs are expensive. It's important that every interaction is productive.
Chatbots or other LLMs that are public-facing can be subject to many different attacks, including 
[prompt injection attacks](https://www.reddit.com/r/ChatGPT/comments/18lxai7/prompt_injection_challenge_chevrolet_of/).

## LLM Testing Difficulties: Why is testing hard?

1. Unrestricted output - users can give the chatbot any instruction in the form of text
2. Unrestricted input - The LLM outputs human-readable text, turning verifying certain properties into a natural language understanding problems.
3. Original training data and procedure unknown - it's impossible to make claims about what the model has been trained on, and re-training from scratch is not feasible.
4. Inference is expensive - With billions of parameters, any test run is expensive, and some testing techniques become computationally infeasible. 

## Testing Properties: What should be tested?

* Correctness
    * Factual correctness - does the output contain strictly factual information?
    * Stylistic correctness - does the model use a helpful and pleasant tone?
    * Structural correctness - does the model's output follow a certain structure, like JSON or YAML?
* Privacy - Does the LLM leak sensitive or private information?
* Security - Is the LLM able to avoid prompt injection attacks, or other attempts to illicit problematic responses?
* Robustness - Does the LLM's behaviour change when extra spaces are added, or when the input is worded differently?
* Fairness - Are model's outputs fair with respect to gender, race, etc., and is equally helpful to all users?
* Model Relevance - Will the model perform well on the data it will encounter in production? Is it overfit? Underfit?

These properties can be tested in **multiple different ways**. See our [test documentation]() for pre-built tests and examples on how to use them.

## General Strategies

LLM testing is extremely difficult, and at the time of writing there is no agreed upon "best practice".

#### 1. Build a focused test suite

LLM testing can become very expensive very quickly; inference is always expensive and sometimes a second language model is needed to evaluate the first.
As such, **focus on use-cases and failure cases that are mission-critical**, and focus on testing **the limit of the model's capabilities**.
A lot of benchmarks and tests online consist of fairly easy questions that modern LLMs always get right, possibly because the questions were included in training data at some point.
It's also unlikely that there will be a benchmark that tests for exactly the kinds of queries your customers might bring to the LLM.
It's important to curate a set of tests specific to your specific use-case, and review/modify that suite should requirements change.

#### 2. Guardrails

Inevitably, testing will surface undesirable behaviour, and solving these bugs becomes the next objective.
Debugging an LLM itself can be difficult, expensive, and slow.
Traditional finetuning and parameter-efficient finetuning can yield mixed results.
Sometimes, it's more efficient and practical to simply filter malicious prompts before they get to the LLM, and/or filter out poor responses before they get to the customer.
For example, a chatbot at a car dealership could have a simple NLP model to classify a user's query as relevant or irrelevant to the vehicles for sale.
In the case of the irrelevant prompt, the chatbot doesn't employ an LLM to respond, and simply states that the query is irrelevant.
Likewise, a chatbot could have its response naively checked for blacklisted words, and abort if the response would have contained foul language.
We call these checks on the input/output of the model "guardrails", and though they aren't as elegant as retraining the model, they can be quite effective.




