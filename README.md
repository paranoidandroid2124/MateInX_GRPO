# ♟️ CoTChess ♟️

<p align="center">
  <img src="AlphaChess.png" alt="AlphaChess Project Banner" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" alt="Project Status"/>
  <img src="https://img.shields.io/badge/Python-3.9+-blue" alt="Python Version"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/> <br/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Library-Transformers-blueviolet" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Library-TRL-critical" alt="TRL"/>
</p>

### Bridging the "Knowing-Doing Gap" in LLM Chess Agents with Chain-of-Thought and Reinforcement Learning

**CoTChess** investigates a common challenge: Large Language Models (LLMs) understand complex game rules, like in chess, yet often fail to apply this knowledge effectively in actual play. This is known as the "Knowing-Doing Gap" (KD Gap) [[1](https://arxiv.org/abs/2504.16078), [2](https://arxiv.org/abs/2504.20073)]. 

Our project explores how **Chain-of-Thought (CoT) prompting** combined with **Reinforcement Learning (GRPO)** can empower LLMs to play chess with greater reliability and strategic depth. We delve into behavioral patterns and neural representations to understand and bridge this gap.

---
## ✨ Team AlphaChess

*   **[Cheol Hun Yeom]** - Team Leader ([drhunny1@gmail.com](mailto:drhunny1@gmail.com))
*   **[In Soo Kim]**
*   **[Ye Rin Hong]**
*   **[Sung Woo Cho]**
*   **[Min Ho Kim]**

---
## 🎯 Research Objectives and Key Questions

**Objective:** Understand the Knowing-Doing Gap (KD Gap) in LLMs playing chess and evaluate how chain-of-thought (CoT) prompting and reinforcement learning fine-tuning (RLFT) influence the reduction of the gap.

**Key Questions:**

*   How does the KD Gap manifest in chess's strategic, sparse-reward setting?
*   What factors underlie the KD Gap, and how do CoT prompting and RLFT address them?
*   Does RLFT for chess induce catastrophic forgetting of domains outside chess, and if yes, how can this be prevented?
*   How do internal representations differ between reasoning (CoT) and action (move), and does RLFT improve their alignment?
*   Does increased test-time compute (e.g., Budget Forcing) improve move quality and narrow the KD Gap?

---
## ✨ Key Features

*   **Custom GRPO Trainer**: Implements a `CustomChessGRPOTrainer` inheriting from `trl.GRPOTrainer` tailored for chess puzzle trajectories.
*   **Chess-Specific Reward Model**: A `ChessReward` class to calculate rewards based on move correctness, format validity, and UCI compliance.
*   **Trajectory-Based Learning**: Executes and learns from multi-step chess puzzle trajectories.
*   **LoRA Integration**: Supports LoRA (Low-Rank Adaptation) for efficient fine-tuning of large language models like `Qwen/Qwen3-0.6B`.
*   **Dynamic Prompting**: Utilizes a `chess_prompt_formatter` to create context-aware prompts for the LLM, incorporating board state, themes, and previous interactions.
*   **Board State Management**: Employs `python-chess` for board representation, move generation, and validation via `simple_context_updater`.
*   **Activation Extraction Scripts**: (Planned) Scripts to extract hidden-state activations from chosen layers and token positions for internal representation analysis.

---
## 🗺️ Research Plan Overview

This project is planned over approximately six weeks, encompassing the following key phases:

<details>
<summary><strong>Task 1: Environment Setup, Baseline and CoT Design</strong></summary>

*   Configure a chess environment (`python-chess`, potentially with `OpenAI-Gym` compatibility) and connect a pre-trained LLM (e.g., `Qwen3` series) to take board inputs and produce moves.
*   Measure zero-shot performance against a weak Stockfish engine.
*   Add scripts to extract hidden-state activations.
*   Draft CoT prompts for chess reasoning (situation analysis, move candidates, evaluation).
*   *Deliverables: Baseline performance data, activation-extraction scripts, CoT prompt templates.*
</details>

<details>
<summary><strong>Task 2: RLFT Pipeline and Initial Experiments</strong></summary>

*   Build an RLFT pipeline (e.g., GRPO, PPO) with chess-based rewards.
*   Prepare benchmarks for catastrophic forgetting (CF).
*   Run short RLFT tests for stability and longer runs against Stockfish, collecting logs and checkpoints.
*   *Deliverables: RLFT code with CoT, initial training logs and checkpoints, CF evaluation setup.*
</details>

<details>
<summary><strong>Task 3: Behavioral Evaluation and Comparison</strong></summary>

*   Define quantitative/qualitative metrics for the KD Gap.
*   Compare the base model with RLFT agents on these metrics.
*   Test inference-time compute variations (e.g., Budget Forcing) and optional no-CoT RLFT.
*   Assess general NLP/math benchmarks to detect CF.
*   *Deliverables: Behavioral metric results, CF evaluation results, initial charts/tables.*
</details>

<details>
<summary><strong>Task 4: Internal Representation Analysis</strong></summary>

*   Collect hidden activations during CoT generation and move output for each agent.
*   Analyze representational similarity (e.g., CKA) and trajectories to compare "Knowing" versus "Doing" states.
*   Evaluate how RLFT alters these representations and their alignment.
*   *Deliverables: Activation data, analysis scripts, quantitative comparison of representations, and interpretations.*
</details>

<details>
<summary><strong>Task 5: Synthesis</strong></summary>

*   Summarize results, insights, and limitations.
*   Draft a paper (Abstract, Introduction, Methods, Results, Discussion) and prepare a presentation.
*   Organize code and data for release.
*   *Deliverables: Final report, paper draft, presentation slides, and codebase.*
</details>

---
## 🛠️ Libraries

| Category                  | Library                                                       | Purpose                                                                 |
| ------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **RL/LLM**                | `trl`                                                         | For GRPO implementation                                                 |
|                           | `transformers`                                                | For LLM models (e.g., `Qwen/Qwen3-0.6B`) and tokenization             |
|                           | `peft`                                                        | For LoRA implementation                                                 |
|                           | `accelerate`                                                  | For distributed training and hardware acceleration                    |
|                           | `torch`                                                       | As the primary deep learning framework                                  |
| **Chess Environment**     | `python-chess`                                                | For chess logic, board states, and move handling                      |
| **Data Handling & Utilities** | `datasets`                                                    | (Potentially for loading/managing chess puzzle datasets - currently uses a dummy dataset) |
|                           | `regex`                                                       | For parsing LLM outputs                                                 |
|                           | `json`                                                        | For handling JSON formatted LLM outputs                                 |

---
## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   An environment with PyTorch installed (preferably with CUDA support for GPU acceleration).

### 2. Clone the Repository

```bash
git clone https://github.com/Mini-Aiffelthon/CoTChess.git
cd CoTChess
```

### 3. Set up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---
## ▶️ How to Run

1.  Ensure your virtual environment is activated and dependencies are installed (see [Getting Started](#-getting-started)).
2.  Configure your training parameters in the `config/` directory (e.g., `config/training_config.yaml` or similar).
3.  Run the main training script from the project root directory:
    ```bash
    python main.py
    ```
4.  Experiment results, logs, and model checkpoints will typically be saved to a directory specified in the configuration (e.g., `results/` or `output/`).
5.  The `notebooks/` directory may contain notebooks for data exploration, analysis, or specific experiments.

**Note**: Ensure your dataset (if not using the dummy dataset) is correctly prepared and paths are configured as expected by the scripts.

---
## 📂 Code Structure Overview

```
CoTChess/
├── AlphaChess.png
├── config/                   
│   └── default.yaml          # Default configuration (model params, training settings, paths)
├── main.py                   # Main script to run training and experiments
├── notebooks/                
│   └── chess-grpo-refactoring.ipynb # Notebook for development, refactoring, or specific analysis
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── reward/                   
│   ├── __init__.py
│   └── chess_reward.py       # Defines the ChessReward class for calculating move rewards
├── trainers/                 
│   ├── __init__.py
│   └── custom_grpo_trainer.py # Implements the CustomChessGRPOTrainer for trajectory learning
├── tests/                    
│   └── test_reward.py        # Unit tests for the reward calculation logic
├── utils/                    
│   ├── __init__.py
│   ├── context.py            # Utilities for managing chess board context (e.g., FEN updates)
│   ├── data.py               # Helper functions for dataset loading and processing
│   ├── json_utils.py         # Utilities for handling JSON parsing from LLM outputs
│   ├── patterns.py           # Regular expression patterns (e.g., for UCI move validation)
│   └── prompting.py          # Functions for formatting prompts sent to the LLM
└── .gitignore
```

<details>
<summary><strong><code>main.py</code></strong></summary>
The main entry point for orchestrating training and evaluation pipelines.
</details>

<details>
<summary><strong><code>config/</code></strong></summary>
Contains configuration files, primarily <code>default.yaml</code>, which holds default settings for model parameters, training configurations (like learning rates, batch sizes), and important paths. This configuration is typically loaded and utilized by <code>main.py</code> to set up experiments.
</details>

<details>
<summary><strong><code>trainers/</code></strong></summary>
Houses the core training logic. The key file is <code>custom_grpo_trainer.py</code>, which implements the <code>CustomChessGRPOTrainer</code> class. This trainer manages the Generative Reward Policy Optimization (GRPO) learning process, including:
<ul>
  <li>Generating chess puzzle trajectories.</li>
  <li>Calculating log probabilities of the LLM's responses.</li>
  <li>Performing optimization steps to update the model.</li>
</ul>
</details>

<details>
<summary><strong><code>reward/</code></strong></summary>
Includes modules for defining and calculating rewards for the LLM's actions. The central component is <code>chess_reward.py</code>, which defines the <code>ChessReward</code> class. This class is responsible for assessing the LLM's performance based on:
<ul>
  <li>Correctness of the predicted chess move.</li>
  <li>Validity of the output format (e.g., JSON structure).</li>
  <li>Compliance with UCI (Universal Chess Interface) standards for moves.</li>
  <li>Other chess-specific criteria.</li>
</ul>
</details>

<details>
<summary><strong><code>utils/</code></strong></summary>
A collection of helper scripts and utility functions to support various parts of the project:
<ul>
  <li><code>context.py</code>: Manages and updates the chess board state (typically FEN strings) as moves are made during a trajectory.</li>
  <li><code>data.py</code>: Provides functions for loading, preprocessing, or batching the chess puzzle data (e.g., handling the <code>dummy_dataset</code> or a more extensive custom dataset).</li>
  <li><code>json_utils.py</code>: Contains utilities to reliably extract and parse JSON content from the LLM's textual outputs, which is crucial for interpreting structured predictions.</li>
  <li><code>patterns.py</code>: Stores compiled regular expression patterns, for example, to validate if a generated move string adheres to the correct UCI format.</li>
  <li><code>prompting.py</code>: Includes functions like <code>chess_prompt_formatter</code> to dynamically create the input prompts fed to the LLM. These prompts are often based on the current board state, puzzle themes, and the history of interaction within a trajectory.</li>
</ul>
</details>

<details>
<summary><strong><code>notebooks/</code></strong></summary>
Contains Jupyter Notebooks such as <code>chess-grpo-refactoring.ipynb</code>. These notebooks are likely used for:
<ul>
  <li>Initial development and prototyping of new features.</li>
  <li>Testing and debugging refactored code components.</li>
  <li>Performing specific data analyses, result visualizations, or ad-hoc experiments.</li>
</ul>
</details>

<details>
<summary><strong><code>tests/</code></strong></summary>
Includes unit tests to ensure code correctness and reliability. For example, <code>test_reward.py</code> would contain tests specifically for the <code>reward/chess_reward.py</code> module, verifying its calculations under various scenarios.
</details>

*   `requirements.txt`: Lists all necessary Python packages for the project.
*   `README.md`: This file - providing an overview, setup, and documentation.
*   `AlphaChess.png`: The project logo.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

---
## �� Expected Outcomes

1.  **Key Insights**:
    *   A clearer picture of how LLMs know rules or strategies but fail to apply them in chess.
    *   Assessment of CoT prompting and RLFT for reducing this gap.
    *   Simple metrics to compare reasoning steps with chosen moves.
2.  **Practical Benefits**:
    *   Guidance for creating more consistent, reliable LLM agents in decision-making tasks.
    *   An open-source codebase and data to support further work on LLM agency.

---
## 🔮 Future Work (Aligned with Research Plan)

This project's future work is closely tied to the **Research Plan Overview** and aims to address the **Key Research Questions**. Key areas include:

*   **Dataset Integration**: Replace `dummy_dataset` with a comprehensive chess puzzle dataset (Task 2).
*   **Activation Extraction Implementation**: Fully develop and integrate scripts for extracting hidden-state activations (Task 1).
*   **Systematic KD Gap Analysis**: Implement and utilize the defined quantitative/qualitative metrics for KD Gap evaluation (Task 3).
*   **Catastrophic Forgetting Benchmarking**: Conduct thorough CF evaluations using established NLP/math benchmarks (Task 2 & 3).
*   **Internal Representation Analysis**: Perform in-depth analysis of hidden activations (e.g., CKA) to compare "Knowing" vs. "Doing" states and the impact of RLFT (Task 4).
*   **Inference-Time Compute Experiments**: Systematically test strategies like Budget Forcing (Task 3).
*   **Advanced CoT & RL Integration**: Explore more sophisticated methods for integrating CoT within the RL loop.
*   **Results Synthesis & Dissemination**: Complete project deliverables including a final report, paper draft, and presentation (Task 5).

---
## 📚 References

1.  Schmied, T., Bornschein, J., Grau-Moya, M., Wulfmeier, R., & Pascanu, R. (2025). *LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities*. arXiv preprint arXiv:2504.16078. URL: [https://arxiv.org/abs/2504.16078](https://arxiv.org/abs/2504.16078)
2.  Wang, Z., Wang, K., Wang, Q., Zhang, P., Li, L., Yang, Z., Yu, K., Nguyen, M. N., Liu, L., Gottlieb, E., Lam, M., Lu, Y., Cho, K., Wu, J., Fei-Fei, L., Wang, L., Choi, Y., & Li, M. (2025). *RaGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*. arXiv preprint arXiv:2504.20073. URL: [https://arxiv.org/abs/2504.20073](https://arxiv.org/abs/2504.20073)

---

Happy Chess Playing and Researching! 🧠♟️
