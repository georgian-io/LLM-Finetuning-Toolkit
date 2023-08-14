#!/bin/bash

dialogues=(
    "Classify the following sentence that is delimited with triple backticks. Sentence: Alice: Hey, how's it going? Bob: Not too bad, just working on this project. Alice: Oh, cool! Which project is it? Bob: It's the new website for the client downtown. Class: "
    "Classify the following sentence that is delimited with triple backticks. Sentence: Emily: Did you catch that new movie everyone's talking about? David: Yeah, I saw it last weekend. It was fantastic! Emily: I've been wanting to watch it. Is it worth the hype? David: Absolutely, the storyline and visuals are impressive.Class:  "
    "Classify the following sentence that is delimited with triple backticks. Sentence: Sarah: Have you decided where to go for the summer vacation? Jake: We're thinking about a beach resort in Spain. Sarah: Spain sounds amazing! When are you planning to go? Jake: Hopefully in August, when the weather is perfect. Class:  "
    "Classify the following sentence that is delimited with triple backticks. Sentence: Linda: I heard you're taking up painting as a hobby now? Michael: Yes, I thought it could be a great way to unwind. Linda: That's wonderful! What kind of things do you paint? Michael: Mostly landscapes and abstract art at the moment.Class:  "
    "Classify the following sentence that is delimited with triple backticks. Sentence: Sophia: Have you tried the new Italian restaurant downtown? Oliver: Not yet, but I've heard it's quite good. Have you been? Sophia: Yes, I went there last week. The pasta was delicious! Oliver: I'll definitely have to check it out then.Class:  "
    "Classify the following sentence that is delimited with triple backticks. Sentence: Grace: I can't believe the semester is almost over already. Daniel: Tell me about it. Time really flew by this time. Grace: I'm looking forward to the break. Any plans for summer? Daniel: Just some relaxation and catching up on reading.Class:  "
)
random_string=${dialogues[$RANDOM % ${#dialogues[@]}]}

echo "{\"inputs\":\"$random_string\"}" > "$1"

./vegeta attack -duration=1s -rate=$2/1s -targets=target.list | ./vegeta report --type=text >> $3