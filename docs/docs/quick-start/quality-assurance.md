---
sidebar_position: 4
---

# Quality Assurance

Once the model is trained, it’s crucial to verify its readiness for production. We offer Quality Assurance testing specifically tailored for Language Model applications. This approach is distinct from conventional testing methods, as there’s currently no direct means of ensuring that a fine-tuned model meets enterprise standards. Moreover, developers have the flexibility to integrate their own tests into the process.

## Available Tests

### Generation Property

#### Generation Length

- **Function**: [`LengthTest`](/docs/api_reference/main-classes/qa#length-test)
- **Description**: Determines the length of the summarized output and the input sentence. The output length is expected to exceed the input length, aligning with the specific use case.

#### POS Composition

- **Description**: Analyzes the grammar of the generated output, focusing on:
  - [Verb Percentage](/docs/api_reference/main-classes/qa#verb-composition): Indicates the proportion of verbs present.
  - [Adjective Percentage](/docs/api_reference/main-classes/qa#adjective-composition): Indicates the proportion of adjectives present.
  - [Noun Percentage](/docs/api_reference/main-classes/qa#noun-composition): Indicates the proportion of nouns present.

### Word Similarity

#### Word Overlap

- **Function**: [`WordOverLapTest`](/docs/api_reference/main-classes/qa#word-overlap)
- **Description**: Determines the length of the summarized output and the input sentence. The output length is expected to exceed the input length, aligning with the specific use case.

#### ROUGE Score

- **Function**: [`RougeScore`](/docs/api_reference/main-classes/qa#rouge-score)
- **Description**: Computes the Rouge score for the output, providing insight into the quality of summarization.

### Embedding Similarity

#### Jaccard Similarity

- **Function**: [`JaccardSimilarity`](/docs/api_reference/main-classes/qa#jaccard-similarity)
- **Description**: Calculates similarity by encoding inputs and outputs.

#### Dot Product (Cosine) Similarity

- **Function**: [`DotProductSimilarity`](/docs/api_reference/main-classes/qa#dot-product-similarity)
- **Description**: Computes the dot product between the encoded inputs and outputs
