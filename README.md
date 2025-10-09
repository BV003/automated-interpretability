# Automated interpretability

### 代码结构

```
.
├── demos
│   ├── explain_puzzles.ipynb  #演示如何为人工设计的 “神经元谜题” 生成解释。谜题包含预设的神经元激活模式和真实解释，用于验证解释生成逻辑的准确性。
│   ├── generate_and_score_explanation.ipynb  # 展示基于激活数据（ActivationRecord）生成解释，并通过模拟和打分评估解释质量的完整流程。
│   └── generate_and_score_token_look_up_table_explanation.ipynb  #示例如何基于 “高激活 token 列表” 生成解释，并进行打分。
├── neuron_explainer
│   ├── activations
│   │   ├── __init__.py
│   │   ├── activation_records.py  # 提供激活数据的格式化工具
│   │   ├── activations.py         # 定义核心数据结构
│   │   └── token_connections.py   # 定义神经元与 token 的关联数据结构,存储高激活 token 及其平均激活值，用于基于 token 列表生成解释。
│   ├── api_client.py         # 封装与大语言模型 API（如 GPT-4）的交互逻辑，处理请求发送、并发控制和缓存，是解释生成和模拟的底层依赖。
│   ├── azure.py            # 提供与 Azure 云存储的交互工具，用于加载公开数据集（如神经元激活数据、预生成的解释）。
│   ├── explanations
│   │   ├── __init__.py
│   │   ├── calibrated_simulator.py
│   │   ├── explainer.py   # 实现解释生成类：基于 token - 激活对生成解释，或者基于高激活 token 列表生成解释。
│   │   ├── explanations.py # 定义解释相关的数据结构
│   │   ├── few_shot_examples.py  # 定义少样本示例集，用于构建解释生成和模拟的提示词
│   │   ├── prompt_builder.py  # 构建提示词的工具类，支持不同格式
│   │   ├── puzzles.json # 定义 “神经元谜题” 的数据结构（Puzzle），包含预设的激活模式、真实解释和错误解释，用于测试解释生成逻辑。
│   │   ├── puzzles.py
│   │   ├── scoring.py  # 实现评分逻辑，核心功能
│   │   ├── simulator.py # 模拟器，实现激活模拟类
│   │   ├── test_explainer.py  # 单元测试文件，验证解释生成和模拟逻辑的正确性
│   │   ├── test_simulator.py
│   │   └── token_space_few_shot_examples.py
│   └── fast_dataclasses    # 提供高效的 dataclass 序列化 / 反序列化工具
│       ├── __init__.py
│       ├── fast_dataclasses.py
│       └── test_fast_dataclasses.py
└── setup.py                 # 项目配置文件，定义依赖和打包信息

```

## 复现构建baseline

