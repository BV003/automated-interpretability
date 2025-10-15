import os
import asyncio

# 从 .env 文件加载环境变量（可选）
from dotenv import load_dotenv
load_dotenv()

from neuron_explainer.activations.activations import ActivationRecordSliceParams, load_neuron
from neuron_explainer.activations.token_connections import load_token_lookup_table_connections_of_neuron
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import TokenSpaceRepresentationExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator


async def main():
    EXPLAINER_MODEL_NAME = "gpt-4"
    SIMULATOR_MODEL_NAME = "text-davinci-003"
    layer_index = 9
    neuron_index = 6236

    # Load a token lookup table.
    token_lookup_table = load_token_lookup_table_connections_of_neuron(layer_index, neuron_index)

    # Load a neuron record.
    neuron_record = load_neuron(layer_index, neuron_index)

    # Grab the activation records we'll need.
    slice_params = ActivationRecordSliceParams(n_examples_per_split=5)
    valid_activation_records = neuron_record.valid_activation_records(
        activation_record_slice_params=slice_params
    )

    # Generate an explanation for the neuron.
    explainer = TokenSpaceRepresentationExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        prompt_format=PromptFormat.HARMONY_V4,
        max_concurrent=1,
    )
    explanations = await explainer.generate_explanations(
        tokens=token_lookup_table.tokens,
        num_samples=1,
    )
    assert len(explanations) == 1
    explanation = explanations[0]
    print(f"{explanation=}")

    # Simulate and score the explanation.
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