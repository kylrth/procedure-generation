from dataset import Procedure, format_steps
from systems import Model
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import re
import asyncio
from evaluation.heuristic import Heuristic

class Overall_Score(Heuristic):
    model: Model

    def __init__(self, model: Model):
        self.model = model


    def prepare_prompt(self, gold_steps: str, proc_steps: str, output: str):
        prompt = [
            SystemMessage(content="[Instruction]\
Please act as an impartial judge and evaluate the quality of the procedure provided by an AI assistant to achieve the user goal displayed below.\
For this evaluation, you should primarily evaluate if the assistant answer provides the same answer and information as the following Ground truth. If the Ground truth provides more information than the assistant answer, it should be penalized; if the ground truth provides information which is not relevant to the user goal than the assistant answer, it should not be penalized; if the assistant answer provides more information, you should evaluate if this information is relevant and correct.\
Do not penalize differences in the sentences' structure, grammar, wording, but focus on the facts and suggestions being made by the assistant; if the assistant answer leads to the same user interpretation and action than the ground truth, then the score should be high.\
However, if key information (like a number referring to an identifier, amount, names, etc.) that is found in the ground truth differs in the assistant answer, it should be penalized, mostly if it changes how the answer is interpreted, wrt the question.\
\
Compare the assistant answer and the Ground truth, then provide a short explanation on your analysis of the assistant answer quality based on the comparison. Be as objective as possible. Based on your explanation, you must rate the response on a scale of 0 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Print this rating at the END only.\
\
As a guideline for scoring:\
[[0]]: the answer is completely contrary to the ground truth and is incorrect.\
[[2]]: the answer largely differs from the ground truth, the important elements found in the ground truth are not present in the assistant answer.\
[[5]]: the answer does not provide all the elements found in the ground truth and some of the elements added by the assistant answer may be incorrect.\
[[8]]: the answer provides most of the elements found in the ground truth, the differences are mainly wording but represent the same facts and suggestions. The differences are relevant and correct.\
[[10]]: the answer is the same as the ground truth. There may be very small wording differences but they mean exactly the same and use the exact same facts and references.\n"),
            HumanMessage(content=f"\n[User Goal]\nProvide a procedure to accomplish: {output}\n\n[The Start of the Ground truth]\n{gold_steps}\n[The End of the Ground truth]\n\n[Assistant Answer]\n{proc_steps}\n\nYour answer should begin with \"Here is my analysis of the comparsion between ground truth and the assistant's answer:\n1.\"")
            ]
        
        
#         prompt = [
#             SystemMessage(content="[Instruction]\
# Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.\
# For this evaluation, you should primarily evaluate if the assistant answer provides the same answer and information as the following Ground truth. If the Ground truth provides more information than the assistant answer, it should be penalized; if the assistant answer provides more information, you should evaluate if this information is relevant and correct.\
# Do not penalize differences in the sentences' structure, grammar, wording, but focus on the facts and suggestions being made by the assistant; if the assistant answer leads to the same user interpretation and action than the ground truth, then the score should be high.\
# However, if key information (like a number referring to an identifier, amount, names, etc.) that is found in the ground truth differs in the assistant answer, it should be penalized, mostly if it changes how the answer is interpreted, wrt the question.\
# \
# Begin your evaluation by providing a comparison between the assistant answer and the Ground truth, then provide a short explanation on your analysis of the assistant answer quality based on the previous comparison. Be as objective as possible. Based on your explanation, you must rate the response on a scale of 0 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Print this rating at the END only.\
# As a guideline for scoring:\
# [[0]]: the answer is completely contrary to the ground truth and is incorrect.\
# [[2]]: the answer largely differs from the ground truth, the important elements found in the ground truth are not present in the assistant answer.\
# [[5]]: the answer does not provide all the elements found in the ground truth and some of the elements added by the assistant answer may be incorrect.\
# [[8]]: the answer provides most of the elements found in the ground truth, the differences are mainly wording but represent the same facts and suggestions. The differences are relevant and correct.\
# [[10]]: the answer is the same as the ground truth. There may be very small wording differences but they mean exactly the same and use the exact same facts and references."),
#             HumanMessage(content=f"\n[The Start of the Ground truth]\n{gold_steps}\n[The End of the Ground truth]\n[Assistant Answer]\n{proc_steps}"),
#             ]
    
        return prompt

    def resultParser(self, sentence):
        if not sentence:
            return -2
        match = re.findall(r'\s*\[\[(\d+)\]\]\s*$', sentence)
        if not match :
            match = re.findall(r'\s*\[\[(\d+)\]\]\s*', sentence)
            if not match :
                match = re.search(r'\s*(N\/A)\s*$', sentence)
                if not match :
                    return -3
                else :
                    return 1

        return int(match[-1])

    async def aevaluate(self, gold: Procedure, generated: list[str]):
        prompt = self.prepare_prompt(gold.format_steps(), format_steps(generated), gold.output)
        print("Prompt prepared")
        answer = (await self.model.agenerate(prompt))[0]
        if answer.find('[') == -1:
            print("Generating the response again")
            answer = (await self.model.agenerate(prompt))[0]
        print("Got model response for overall score")
        # import pdb; pdb.set_trace()
        score = self.resultParser(answer)
        print("Returning the achieved score!")
        return score
    