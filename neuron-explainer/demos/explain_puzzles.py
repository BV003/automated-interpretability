import os
import asyncio
from dotenv import load_dotenv
load_dotenv("/mnt/d/home/Du/automated-interpretability/.env")
print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.puzzles import PUZZLES_BY_NAME




async def main():
    # 设置OpenAI API密钥
    EXPLAINER_MODEL_NAME = "gpt-4"

    # 初始化解释器
    explainer = TokenActivationPairExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )

    # 遍历所有谜题并生成解释
    for puzzle_name, puzzle in PUZZLES_BY_NAME.items():
        print(f"{puzzle_name=}")
        puzzle_answer = puzzle.explanation
        
        # 生成谜题的解释
        explanations = await explainer.generate_explanations(
            all_activation_records=puzzle.activation_records,
            max_activation=calculate_max_activation(puzzle.activation_records),
            num_samples=1,
        )
        
        assert len(explanations) == 1
        model_generated_explanation = explanations[0]
        
        print(f"{model_generated_explanation=}")
        print(f"{puzzle_answer=}\n")


if __name__ == "__main__":
    asyncio.run(main())