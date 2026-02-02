MTA_SYSTEM_PRMOPT = """
We want to evaluate the quality of the responses provided by AI assistants to the user question displayed below. 
For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the evaluation plan.
""".strip()

MTA_FIRST_USER_PROMPT = """
[User Question]
{question}
[End of User Question]
""".strip()

RA_SYSTEM_PRMOPT="""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should be performed by following the provided evaluation plan step-by-step. Avoid copying the plan when doing the evaluation. 
Please also only stick to the given plan and provide explanation of how the plan is executed to compare the two responses. 
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
After providing your evaluation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.
""".strip()

RA_FIRST_USER_PROMPT="""
[User Question]
{question}
[End of User Question]
[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
[The Start of Evaluation Plan]
{instruction}
[The End of Evaluation Plan]
""".strip()