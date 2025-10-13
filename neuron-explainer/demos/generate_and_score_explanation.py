import os
import asyncio

# 从 .env 文件加载环境变量（可选）
from dotenv import load_dotenv
load_dotenv()



from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator


async def main():
    EXPLAINER_MODEL_NAME = "gpt-4"
    SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"


    # 加载神经元记录
    neuron_record = load_neuron(9, 6236)

    # 提取激活记录
    slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
    train_activation_records = neuron_record.train_activation_records(
        activation_record_slice_params=slice_params
    )
    valid_activation_records = neuron_record.valid_activation_records(
        activation_record_slice_params=slice_params
    )

    # 使用 TokenActivationPairExplainer 生成解释
    explainer = TokenActivationPairExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )
    explanations = await explainer.generate_explanations(
        all_activation_records=train_activation_records,
        max_activation=calculate_max_activation(train_activation_records),
        num_samples=1,
    )

    assert len(explanations) == 1
    explanation = explanations[0]
    print(f"explanation={explanation}")

    # 模拟并评分
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=1,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )

    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    print(f"score={scored_simulation.get_preferred_score():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
