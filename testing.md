## LLM Testing Guidebook

Below, we outline an initial guide to testing LLMs.
LLMs are some of the hardest software systems to test, and have some unique challenges when compared to other ML systems.

## Motivation

Companies can be held [liable for their chatbot's outputs](https://www.cbc.ca/news/canada/british-columbia/air-canada-chatbot-lawsuit-1.7116416).
Additionally, LLMs are expensive. It's important that every interaction is productive.
Chatbots or other LLMs that are public facing can be subject to many different attacks, including 
[prompt injection attacks](https://www.reddit.com/r/ChatGPT/comments/18lxai7/prompt_injection_challenge_chevrolet_of/).

## LLM Testing Difficulties

LLM testing is extremely difficult, and at the time of writing there is no agreed upon "best practice".

1. Unrestricted output - users can give the chatbot any instruction in the form of text
2. Unrestricted input - The LLM outputs human-readable text, turning verifying certain properties into a natural language understanding problems.
3. Original training data and procedure unknown - it's impossible to make claims about what the model has been trained on, and re-training from scratch is not feasible.
4. Inference is expensive - With billions of parameters, any test run is expensive, and some testing techniques become computationally infeasible. 

## Testing Properties

* Correctness
    * Factual correctness - does the output contain strictly factual information?
    * Stylistic correctness - does the model use a helpful and pleasant tone?
    * Structural correctness - does the model's output follow a certain structure, like JSON or YAML?
* Privacy - Does the LLM leak sensitive or private information?
* Security - Is the LLM able to avoid prompt injection attacks?
* Robustness - Does the LLM's behaviour change when extra spaces are added, or when the input is worded differently?
* Fairness - The model's outputs are fair, with respect to gender, race, etc. and is equally helpful to all users
* Model Relevance - Will the model perform well on the data it will encounter in production? Is it overfit? Underfit?

## Where to test

* Data Testing
* Learning Program (Model) Testing

## Debug and Repair

How do we fix our LLM, if it has undesirable behaviour?

1. Prompt - sometimes 
2. External Guardrails
3. Finetuning




