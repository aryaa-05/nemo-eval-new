from typing import Dict, Any, Tuple

class LLMJudge:
    @staticmethod
    def build_pairwise_prompt(question: str, response_a: str, response_b: str, reference: str = None) -> str:
        ref_clause = f"\nReference Answer: {reference}" if reference else ""
        prompt = f"""You are an impartial judge evaluating responses to the following question.
Question: {question}{ref_clause}

Response A:
{response_a}

Response B:
{response_b}

Criteria: Correctness, depth of reasoning, clarity, and conciseness.
Which response is better?
Respond in the following format:
Winner: [[A]], [[B]], or [[Tie]]
Explanation: <brief reasoning>
"""
        return prompt

    @staticmethod
    def parse_judge_winner(judge_output: str) -> str:
        if "[[A]]" in judge_output:
            return "A"
        elif "[[B]]" in judge_output:
            return "B"
        elif "[[Tie]]" in judge_output:
            return "Tie"
        return "Tie"

    @staticmethod
    def evaluate_pairwise_debiased(judge_client, question: str, resp_1: str, resp_2: str, reference: str = None) -> Dict[str, Any]:
        p1 = LLMJudge.build_pairwise_prompt(question, resp_1, resp_2, reference)
        out1 = judge_client.generate(p1)
        w1 = LLMJudge.parse_judge_winner(out1)

        p2 = LLMJudge.build_pairwise_prompt(question, resp_2, resp_1, reference)
        out2 = judge_client.generate(p2)
        w2 = LLMJudge.parse_judge_winner(out2)

        if w1 == "A" and w2 == "B":
            final_winner = "Model_1"
        elif w1 == "B" and w2 == "A":
            final_winner = "Model_2"
        else:
            final_winner = "Tie"

        return {
            "winner": final_winner,
            "forward_winner": w1,
            "reverse_winner": w2,
            "explanation_forward": out1,
            "explanation_reverse": out2
        }
