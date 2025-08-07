model2prompt = {
    "Default": """### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 100. For your rating, only give a number between 0 and 100 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{response}

[Your judgement]""",
    "Selene-1-Mini-Llama-3.1-8B": """You are tasked with evaluating a response based on a given instruction (which may contain an Input) and a scoring rubric and reference answer that serve as the evaluation standard. Provide a comprehensive feedback on the response quality strictly adhering to the scoring rubric, without any general evaluation. Follow this with a score between 1 and 5, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric.
Rather, evaluate the response based on the criteria outlined in the rubric.
(2) You should refer to the provided reference answer as a guide for evaluating the response.
Your reply should strictly follow this format:
**Reasoning:** <Your feedback>
**Result:** <an integer between 1 and 5>
Here is the data:
Instruction:
{prompt}

Response:
{response}

Please provide your evaluation:""",
    "Skywork-Critic-Llama-3.1-8B": """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 100. For your rating, only give a number between 0 and 100 (inclusive), do not use any markdown, and do not put any text after your final rating.

[User Question]
{prompt}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]""",

}