<a name="readme-top"></a>

<h1 align="center">Awesome Synthetic Data Generation</h1>

<div align="center">

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
![Forks](https://img.shields.io/github/forks/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
<a href='https://arxiv.org/pdf/2409.18169'><img src='https://img.shields.io/badge/arXiv-2409.18169-b31b1b.svg'></a>

</div>

<p align="center">
    <b>A comprehensive survey and curated collection of resources on synthetic data generation.</b>
</p>

---

<details>
<summary><strong>📚 Table of Contents (Click to Expand)</strong></summary>

- [1. Methodologies](#1-methodologies)
  - [1.1 Generation-Based Synthesis](#11-generation-based-synthesis)
  - [1.2 Inversion-Based Synthesis](#12-inversion-based-synthesis)
  - [1.3 Simulation-Based Synthesis](#13-simulation-based-synthesis)
  - [1.4 Augmentation-Based Synthesis](#14-augmentation-based-synthesis)
- [2. Applications](#2-applications)
  - [2.1 Data-centric AI](#21-data-centric-ai)
    - [Data Accessibility](#data-accessibility)
    - [Data Refinement](#data-refinement)
  - [2.2 Model-centric AI](#22-model-centric-ai)
    - [General Model Enhancement](#general-model-enhancement)
    - [Domain Model Enhancement](#domain-model-enhancement)
    - [Model Evaluation](#model-evaluation)
  - [2.3 Trustworthy AI](#23-trustworthy-ai)
    - [Privacy](#privacy)
    - [Safety & Security](#safety--security)
    - [Fairness](#fairness)
    - [Interpretability](#interpretability)
    - [Governance](#governance)
  - [2.4 Embodied AI](#24-embodied-ai)
    - [Perception](#perception)
    - [Interaction](#interaction)
    - [Generalization](#generalization)
- [3. Challenges & Future Directions](#3-challenges--future-directions)

</details>

---

## 1. Methodologies

### 1.1 Generation-Based Synthesis

**Synthesis from scratch**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Self-Instruct | 2023 | ACL |
| SynCLR | 2024 | ICML |
| Absolute Zero | 2024 | ICLR |
| DreamTeacher | 2023 | ICCV |

**Synthesis from seeds**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Mosaic | - | - |
| CORE | - | - |
| ALIA | - | - |
| ChatAug | - | - |

**Synthesis from structure**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Co-annotating | - | - |
| ToolCoder | - | - |

**Synthesis with evolution**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Co-annotating | - | - |
| ToolCoder | - | - |

### 1.2 Inversion-Based Synthesis

**Data-space inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| TinyStories | 2023 | ACL |
| Phi-1 | 2023 | arXiv |
| Alpagasus | 2024 | ICLR |
| WizardLM | 2024 | ICLR |

**Latent-space inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Minerva | 2022 | NeurIPS |
| DeepSeek-Prover | 2024 | arXiv |
| WizardCoder | 2023 | ICLR |

### 1.3 Simulation-Based Synthesis

**Agent-based simulation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| TinyStories | 2023 | ACL |
| Phi-1 | 2023 | arXiv |
| Alpagasus | 2024 | ICLR |
| WizardLM | 2024 | ICLR |

**Platform-based simulation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Minerva | 2022 | NeurIPS |
| DeepSeek-Prover | 2024 | arXiv |
| WizardCoder | 2023 | ICLR |

### 1.4 Augmentation-Based Synthesis

**Rule-based augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| TinyStories | 2023 | ACL |
| Phi-1 | 2023 | arXiv |
| Alpagasus | 2024 | ICLR |
| WizardLM | 2024 | ICLR |

**Generative augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Minerva | 2022 | NeurIPS |
| DeepSeek-Prover | 2024 | arXiv |
| WizardCoder | 2023 | ICLR |

---

## 2. Applications

### 2.1 Data-centric AI

#### Data Accessibility

**Zero/Few-shot learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

**Federated learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

**Data-free knowledge distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

**Data-free pruning/quantization**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

**Data-free meta-learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

**Data-free continual learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Phi-1 | 2023 | arXiv |
| SciLitLLM | 2024 | arXiv |
| TRAIT | - | - |
| AnyGPT | - | - |
| Phi-1.5 | 2023 | arXiv |

#### Data Refinement

**Dataset distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| VILA-2 | 2024 | arXiv |

**Dataset purification**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| VILA-2 | 2024 | arXiv |

---

### 2.2 Model-centric AI

#### General Model Enhancement

**General ability**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

#### Domain Model Enhancement

**Reasoning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

**Code**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

**Instruction following**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

**Preference**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

**Reinforcement Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

#### Model Evaluation

**In-context learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Dialogic | - | - |
| MathInstruct | 2024 | arXiv |
| Genixer | - | - |
| Magpie | 2024 | arXiv |
| MMIQC | - | - |

**Synthetic benchmark**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Disco | - | - |
| GPT3Mix | 2021 | arXiv |

---

### 2.3 Trustworthy AI

#### Privacy

**Privacy-preserving learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

**Model inversion attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

#### Safety & Security

**Model stealing attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

**Adversarial defense**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

**Machine unlearning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

#### Fairness

**De-bias learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

**Long-tail learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

#### Interpretability

**Explainable AI**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

#### Governance

**Data watermarking**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LAB | - | - |
| LLM2LLM | 2024 | arXiv |
| GLAN | - | - |

---

### 2.4 Embodied AI

#### Perception

**Visual sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Isaac Lab | 2024 | arXiv |
| SplatSim | - | - |
| Re3Sim | - | - |
| Habitat | 2019 | ICCV |

**Force sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| DeXtreme | 2023 | ICRA |
| ARNOLD | 2023 | ICRA |
| Isaac Lab | 2024 | arXiv |

**Sensor fusion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| SpatialVLM | 2024 | CVPR |
| RT-2 | 2023 | arXiv |
| PaLM-E | 2023 | ICML |
| EmbodiedGPT | 2023 | NeurIPS |

#### Interaction

**Trajectory synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

**Environment synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

**Human behavior synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

#### Generalization

**Cross-embodiment training**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

**Vision-language-action models**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

**Sim-to-real transfer**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| Alpaca | 2023 | Stanford |
| Vicuna | 2023 | LMSYS |
| Orca | 2023 | arXiv |
| Baize | 2023 | EMNLP |
| LLaVA | 2023 | NeurIPS |

---

## 3. Challenges & Future Directions

**Model Collapse**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| d-RLAIF | - | - |
| LLM2LLM | 2024 | arXiv |
| Wizardmath | 2023 | arXiv |
| STaR | 2022 | NeurIPS |
| SciGLM | 2024 | arXiv |
| ChemLLM | 2024 | arXiv |

**Active Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| LLMs4Synthesis | - | - |
| CoRAL | - | - |
| FORD | - | - |
| LTGC | - | - |

**Synthetic Data Evaluation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| DataDreamer | 2024 | arXiv |
| HARMONIC | - | - |

**Multi-Modal Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| PANDA | - | - |
| REGA | - | - |

---

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
