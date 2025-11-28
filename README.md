<a name="readme-top"></a>

<h1 align="center">Awesome Synthetic Data Generation</h1>

<div align="center">

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
![Forks](https://img.shields.io/github/forks/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
<a href='https://arxiv.org/pdf/2409.18169'><img src='https://img.shields.io/badge/arXiv-2409.18169-b31b1b.svg'></a>

</div>

<p align="center">
    <b>A comprehensive survey and curated collection of resources on synthetic data generation, organized by Methodologies, Applications, and Challenges.</b>
</p>

---

<details>
<summary><strong>📚 Table of Contents (Click to Expand)</strong></summary>

- [1. Methodologies (How)](#1-methodologies-how)
  - [1.1 Generation-Based Synthesis](#11-generation-based-synthesis)
  - [1.2 Inversion-Based Synthesis](#12-inversion-based-synthesis)
  - [1.3 Simulation-Based Synthesis](#13-simulation-based-synthesis)
  - [1.4 Augmentation-Based Synthesis](#14-augmentation-based-synthesis)
- [2. Applications (Why & Where)](#2-applications-why--where)
  - [2.1 Data-centric AI](#21-data-centric-ai)
    - [2.1.1 Data Accessibility](#211-data-accessibility)
    - [2.1.2 Data Refinement](#212-data-refinement)
  - [2.2 Model-centric AI](#22-model-centric-ai)
    - [2.2.1 General Model Enhancement](#221-general-model-enhancement)
    - [2.2.2 Domain Model Enhancement](#222-domain-model-enhancement)
    - [2.2.3 Model Evaluation](#223-model-evaluation)
  - [2.3 Trustworthy AI](#23-trustworthy-ai)
    - [2.3.1 Privacy](#231-privacy)
    - [2.3.2 Safety & Security](#232-safety--security)
    - [2.3.3 Fairness](#233-fairness)
    - [2.3.4 Interpretability](#234-interpretability)
    - [2.3.5 Governance](#235-governance)
  - [2.4 Embodied AI](#24-embodied-ai)
    - [2.4.1 Perception](#241-perception)
    - [2.4.2 Interaction](#242-interaction)
    - [2.4.3 Generalization](#243-generalization)
- [3. Challenges & Future Directions](#3-challenges--future-directions)

</details>

---

## 1. Methodologies (How)

### 1.1 Generation-Based Synthesis
*Creating data from scratch, seeds, or structural constraints.*

| Sub-category | Representative Papers/Methods |
|--------------|-------------------------------|
| **Synthesis from scratch** | Self-Instruct, SynCLR, Absolute Zero, DreamTeacher |
| **Synthesis from seeds** | Mosaic, CORE, ALIA, ChatAug |
| **Synthesis from structure** | Co-annotating, ToolCoder |
| **Synthesis with evolution** | Co-annotating, ToolCoder |

### 1.2 Inversion-Based Synthesis
*Recovering input data from model parameters or outputs.*

| Sub-category | Representative Papers/Methods |
|--------------|-------------------------------|
| **Data-space inversion** | TinyStories, Phi-1, Alpagasus, WizardLM |
| **Latent-space inversion** | Minerva, DeepSeek-Prover, WizardCoder |

### 1.3 Simulation-Based Synthesis
*Generating data via multi-agent interactions or physics engines.*

| Sub-category | Representative Papers/Methods |
|--------------|-------------------------------|
| **Agent-based simulation** | TinyStories, Phi-1, Alpagasus, WizardLM |
| **Platform-based simulation** | Minerva, DeepSeek-Prover, WizardCoder |

### 1.4 Augmentation-Based Synthesis
*Transforming existing data to enhance diversity.*

| Sub-category | Representative Papers/Methods |
|--------------|-------------------------------|
| **Rule-based augmentation** | TinyStories, Phi-1, Alpagasus, WizardLM |
| **Generative augmentation** | Minerva, DeepSeek-Prover, WizardCoder |

---

## 2. Applications (Why & Where)

### 2.1 Data-centric AI
*Building the Foundation: Overcoming scarcity and improving quality.*

#### 2.1.1 Data Accessibility
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Zero/Few-shot learning** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |
| **Federated learning** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |
| **Data-free knowledge distillation** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |
| **Data-free pruning/quantization** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |
| **Data-free meta-learning** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |
| **Data-free continual learning** | Phi-1, SciLitLLM, TRAIT, AnyGPT, Phi-1.5 |

#### 2.1.2 Data Refinement
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Dataset distillation** | VILA-2 |
| **Dataset purification** | VILA-2 |

---

### 2.2 Model-centric AI
*Driving the Engine: Enhancing capabilities and evaluation.*

#### 2.2.1 General Model Enhancement
| Goal | Representative Papers/Methods |
|------|-------------------------------|
| **General ability** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |

#### 2.2.2 Domain Model Enhancement
| Goal | Representative Papers/Methods |
|------|-------------------------------|
| **Reasoning** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |
| **Code** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |
| **Instruction following** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |
| **Preference** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |
| **Reinforcement Learning** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |

#### 2.2.3 Model Evaluation
| Goal | Representative Papers/Methods |
|------|-------------------------------|
| **In-context learning** | Dialogic, MathInstruct, Genixer, Magpie, MMIQC |
| **Synthetic benchmark** | Disco, GPT3Mix |

---

### 2.3 Trustworthy AI
*The Guardrails: Ensuring safety, privacy, and fairness.*

#### 2.3.1 Privacy
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Privacy-preserving learning** | LAB, LLM2LLM, GLAN |
| **Model inversion attack** | LAB, LLM2LLM, GLAN |

#### 2.3.2 Safety & Security
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Model stealing attack** | LAB, LLM2LLM, GLAN |
| **Adversarial defense** | LAB, LLM2LLM, GLAN |
| **Machine unlearning** | LAB, LLM2LLM, GLAN |

#### 2.3.3 Fairness
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **De-bias learning** | LAB, LLM2LLM, GLAN |
| **Long-tail learning** | LAB, LLM2LLM, GLAN |

#### 2.3.4 Interpretability
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Explainable AI** | LAB, LLM2LLM, GLAN |

#### 2.3.5 Governance
| Problem Setting | Representative Papers/Methods |
|-----------------|-------------------------------|
| **Data watermarking** | LAB, LLM2LLM, GLAN |

---

### 2.4 Embodied AI
*The Frontier: Bridging digital intelligence with the physical world.*

#### 2.4.1 Perception
| Task | Representative Papers/Methods |
|------|-------------------------------|
| **Visual sensing** | Isaac Lab, SplatSim, Re3Sim, Habitat |
| **Force sensing** | DeXtreme, ARNOLD, Isaac Lab |
| **Sensor fusion** | SpatialVLM, RT-2, PaLM-E, EmbodiedGPT |

#### 2.4.2 Interaction
| Task | Representative Papers/Methods |
|------|-------------------------------|
| **Trajectory synthesis** | Alpaca, Vicuna, Orca, Baize, LLaVA |
| **Environment synthesis** | Alpaca, Vicuna, Orca, Baize, LLaVA |
| **Human behavior synthesis** | Alpaca, Vicuna, Orca, Baize, LLaVA |

#### 2.4.3 Generalization
| Task | Representative Papers/Methods |
|------|-------------------------------|
| **Cross-embodiment training** | Alpaca, Vicuna, Orca, Baize, LLaVA |
| **Vision-language-action models** | Alpaca, Vicuna, Orca, Baize, LLaVA |
| **Sim-to-real transfer** | Alpaca, Vicuna, Orca, Baize, LLaVA |

---

## 3. Challenges & Future Directions

| Challenge | Representative Papers/Methods |
|-----------|-------------------------------|
| **Model Collapse** | d-RLAIF, LLM2LLM, Wizardmath, STaR, SciGLM, ChemLLM |
| **Active Synthesis** | LLMs4Synthesis, CoRAL, FORD, LTGC |
| **Synthetic Data Evaluation** | DataDreamer, HARMONIC |
| **Multi-Modal Synthesis** | PANDA, REGA |

---

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
