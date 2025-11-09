<a name="readme-top"></a>

<h1 align="center">Awesome Synthetic Data Generation</h1>

<div align="center">

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
![Forks](https://img.shields.io/github/forks/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
<a href='https://arxiv.org/pdf/2409.18169'><img src='https://img.shields.io/badge/arXiv-2409.18169-b31b1b.svg'></a>

</div>

<p align="center">
    <b>Curated collection of papers and resources on synthetic data generation, categorized by paradigms, applications, and challenges.</b>
</p>

---

<details>
<summary><strong>📚 Content (click to expand)</strong></summary>

- [1. Paradigms](#1-paradigms)
  - [1.1 AIGC-Based Synthesis](#11-aigc-based-synthesis)
    - [1.1.1 Synthesis from scratch](#111-synthesis-from-scratch)
    - [1.1.2 Synthesis from seeds](#112-synthesis-from-seeds)
    - [1.1.3 Synthesis from structure](#113-synthesis-from-structure)
  - [1.2 Inversion-Based Synthesis](#12-inversion-based-synthesis)
    - [1.2.1 Data-space inversion](#121-data-space-inversion)
    - [1.2.2 Latent-space inversion](#122-latent-space-inversion)
  - [1.3 Simulation-Based Synthesis](#13-simulation-based-synthesis)
    - [1.3.1 Agent-based simulation](#131-agent-based-simulation)
    - [1.3.2 Platform-based simulation](#132-platform-based-simulation)
  - [1.4 Augmentation-Based Synthesis](#14-augmentation-based-synthesis)
    - [1.4.1 Rule-based augmentation](#141-rule-based-augmentation)
    - [1.4.2 Generative augmentation](#142-generative-augmentation)

- [2. Applications](#2-applications)
  - [2.1 Model-centric AI](#21-model-centric-ai)
    - [2.1.1 General Model Enhancement → General ability](#211-general-model-enhancement--general-ability)
    - [2.1.2 Domain Model Enhancement → Reasoning](#212-domain-model-enhancement--reasoning)
    - [2.1.3 Domain Model Enhancement → Code](#213-domain-model-enhancement--code)
    - [2.1.4 Domain Model Enhancement → Instruction following](#214-domain-model-enhancement--instruction-following)
    - [2.1.5 Domain Model Enhancement → Alignment](#215-domain-model-enhancement--alignment)
    - [2.1.6 Domain Model Enhancement → In-context learning](#216-domain-model-enhancement--in-context-learning)
    - [2.1.7 Model Evaluation → Synthetic benchmark](#217-model-evaluation--synthetic-benchmark)
  - [2.2 Data-centric AI](#22-data-centric-ai)
    - [2.2.1 Data Accessibility → Zero/Few-shot learning](#221-data-accessibility--zerofew-shot-learning)
    - [2.2.2 Data Accessibility → Federated learning](#222-data-accessibility--federated-learning)
    - [2.2.3 Data Accessibility → Data-free knowledge distillation](#223-data-accessibility--data-free-knowledge-distillation)
    - [2.2.4 Data Accessibility → Data-free pruning/quantization](#224-data-accessibility--data-free-pruningquantization)
    - [2.2.5 Data Accessibility → Data-free meta-learning](#225-data-accessibility--data-free-meta-learning)
    - [2.2.6 Data Accessibility → Data-free continual learning](#226-data-accessibility--data-free-continual-learning)
    - [2.2.7 Data Refinement → Dataset distillation](#227-data-refinement--dataset-distillation)
    - [2.2.8 Data Refinement → Dataset augmentation](#228-data-refinement--dataset-augmentation)
    - [2.2.9 Data Refinement → Dataset expansion](#229-data-refinement--dataset-expansion)
    - [2.2.10 Data Refinement → Dataset purification](#2210-data-refinement--dataset-purification)
  - [2.3 Trustworthy AI](#23-trustworthy-ai)
    - [2.3.1 Privacy → Privacy-preserving learning](#231-privacy--privacy-preserving-learning)
    - [2.3.2 Safety & Security → Model inversion attack](#232-safety--security--model-inversion-attack)
    - [2.3.3 Safety & Security → Model stealing attack](#233-safety--security--model-stealing-attack)
    - [2.3.4 Safety & Security → Machine unlearning](#234-safety--security--machine-unlearning)
    - [2.3.5 Fairness → De-bias learning](#235-fairness--de-bias-learning)
    - [2.3.6 Fairness → Long-tail learning](#236-fairness--long-tail-learning)
    - [2.3.7 Interpretability → Explainable AI](#237-interpretability--explainable-ai)
    - [2.3.8 Governance → Data watermarking](#238-governance--data-watermarking)
  - [2.4 Embodied AI](#24-embodied-ai)
    - [2.4.1 Sensory Perception Synthesis → Visual Modalities](#241-sensory-perception-synthesis--visual-modalities)
    - [2.4.2 Sensory Perception Synthesis → Proprioceptive & Force Sensing](#242-sensory-perception-synthesis--proprioceptive--force-sensing)
    - [2.4.3 Sensory Perception Synthesis → Multi-modal Sensor Fusion](#243-sensory-perception-synthesis--multi-modal-sensor-fusion)
    - [2.4.4 Action & Behavior Generation → Trajectory Synthesis](#244-action--behavior-generation--trajectory-synthesis)
    - [2.4.5 Action & Behavior Generation → Task & Environment Generation](#245-action--behavior-generation--task--environment-generation)
    - [2.4.6 Action & Behavior Generation → Human Behavior Synthesis](#246-action--behavior-generation--human-behavior-synthesis)
    - [2.4.7 Policy Learning & Generalization → Cross-embodiment Training](#247-policy-learning--generalization--cross-embodiment-training)
    - [2.4.8 Policy Learning & Generalization → Vision-Language-Action Models](#248-policy-learning--generalization--vision-language-action-models)
    - [2.4.9 Policy Learning & Generalization → Sim-to-Real Transfer](#249-policy-learning--generalization--sim-to-real-transfer)
  - [2.5 Others](#25-others)
    - [2.5.1 Domain-specific Applications → Autonomous driving](#251-domain-specific-applications--autonomous-driving)
    - [2.5.2 Domain-specific Applications → Finance](#252-domain-specific-applications--finance)
    - [2.5.3 Domain-specific Applications → Medical](#253-domain-specific-applications--medical)
    - [2.5.4 Domain-specific Applications → Law](#254-domain-specific-applications--law)
    - [2.5.5 Domain-specific Applications → Education](#255-domain-specific-applications--education)
    - [2.5.6 Structure-specific Applications → Time series](#256-structure-specific-applications--time-series)
    - [2.5.7 Structure-specific Applications → Tabular](#257-structure-specific-applications--tabular)
    - [2.5.8 Structure-specific Applications → Graph](#258-structure-specific-applications--graph)

- [3. Challenges & Future Directions](#3-challenges--future-directions)
  - [3.1 Model Collapse](#31-model-collapse)
  - [3.2 Active Synthesis](#32-active-synthesis)
  - [3.3 Synthetic Data Evaluation](#33-synthetic-data-evaluation)
  - [3.4 Multi-Modal Synthesis](#34-multi-modal-synthesis)

</details>

---

## 1. Paradigms

### 1.1 AIGC-Based Synthesis

#### 1.1.1 Synthesis from scratch
| Title | Conference | Year | Code |
|-------|------------|------|------|
| T-SciQ | - | - | - |
| ChatGPT-based | - | - | - |

#### 1.1.2 Synthesis from seeds
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Mosaic | - | - | - |
| CORE | - | - | - |
| ALIA | - | - | - |
| ChatAug | - | - | - |

#### 1.1.3 Synthesis from structure
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Co-annotating | - | - | - |
| ToolCoder | - | - | - |

---

### 1.2 Inversion-Based Synthesis

#### 1.2.1 Data-space inversion
| Title | Conference | Year | Code |
|-------|------------|------|------|
| TinyStories | - | - | - |
| Phi-1 | - | - | - |
| Alpagasus | - | - | - |
| WizardLM | - | - | - |

#### 1.2.2 Latent-space inversion
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Minerva | - | - | - |
| DeepSeek-Prover | - | - | - |
| WizardCoder | - | - | - |

---

### 1.3 Simulation-Based Synthesis

#### 1.3.1 Agent-based simulation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| TinyStories | - | - | - |
| Phi-1 | - | - | - |
| Alpagasus | - | - | - |
| WizardLM | - | - | - |

#### 1.3.2 Platform-based simulation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Minerva | - | - | - |
| DeepSeek-Prover | - | - | - |
| WizardCoder | - | - | - |

---

### 1.4 Augmentation-Based Synthesis

#### 1.4.1 Rule-based augmentation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| TinyStories | - | - | - |
| Phi-1 | - | - | - |
| Alpagasus | - | - | - |
| WizardLM | - | - | - |

#### 1.4.2 Generative augmentation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Minerva | - | - | - |
| DeepSeek-Prover | - | - | - |
| WizardCoder | - | - | - |

---

## 2. Applications

### 2.1 Model-centric AI

#### 2.1.1 General Model Enhancement

##### General ability
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

##### Reasoning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

##### Code
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

##### Instruction following
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

##### Alignment
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

##### In-context learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Dialogic | - | - | - |
| MathInstruct | - | - | - |
| Genixer | - | - | - |
| Magpie | - | - | - |
| MMIQC | - | - | - |

---

#### 2.1.2 Model Evaluation

##### Synthetic benchmark
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Disco | - | - | - |
| GPT3Mix | - | - | - |

---

### 2.2 Data-centric AI

#### 2.2.1 Data Accessibility

##### Zero/Few-shot learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

##### Federated learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

##### Data-free knowledge distillation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

##### Data-free pruning/quantization
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

##### Data-free meta-learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

##### Data-free continual learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Phi-1 | - | - | - |
| SciLitLLM | - | - | - |
| TRAIT | - | - | - |
| AnyGPT | - | - | - |
| Phi-1.5 | - | - | - |

---

#### 2.2.2 Data Refinement

##### Dataset distillation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| VILA-2 | - | - | - |

##### Dataset augmentation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| VILA-2 | - | - | - |

##### Dataset expansion
| Title | Conference | Year | Code |
|-------|------------|------|------|
| VILA-2 | - | - | - |

##### Dataset purification
| Title | Conference | Year | Code |
|-------|------------|------|------|
| VILA-2 | - | - | - |

---

### 2.3 Trustworthy AI

#### 2.3.1 Privacy

##### Privacy-preserving learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

---

#### 2.3.2 Safety & Security

##### Model inversion attack
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

##### Model stealing attack
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

##### Machine unlearning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

---

#### 2.3.3 Fairness

##### De-bias learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

##### Long-tail learning
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

---

#### 2.3.4 Interpretability

##### Explainable AI
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

---

#### 2.3.5 Governance

##### Data watermarking
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LAB | - | - | - |
| LLM2LLM | - | - | - |
| GLAN | - | - | - |

---

### 2.4 Embodied AI

#### 2.4.1 Sensory Perception Synthesis

##### Visual Modalities
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Isaac Lab | - | - | - |
| SplatSim | - | - | - |
| Re3Sim | - | - | - |
| Habitat | - | - | - |

##### Proprioceptive & Force Sensing
| Title | Conference | Year | Code |
|-------|------------|------|------|
| DeXtreme | - | - | - |
| ARNOLD | - | - | - |
| Isaac Lab | - | - | - |

##### Multi-modal Sensor Fusion
| Title | Conference | Year | Code |
|-------|------------|------|------|
| SpatialVLM | - | - | - |
| RT-2 | - | - | - |
| PaLM-E | - | - | - |
| EmbodiedGPT | - | - | - |

---

#### 2.4.2 Action & Behavior Generation

##### Trajectory Synthesis
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

##### Task & Environment Generation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

##### Human Behavior Synthesis
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

---

#### 2.4.3 Policy Learning & Generalization

##### Cross-embodiment Training
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

##### Vision-Language-Action Models
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

##### Sim-to-Real Transfer
| Title | Conference | Year | Code |
|-------|------------|------|------|
| Alpaca | - | - | - |
| Vicuna | - | - | - |
| Orca | - | - | - |
| Baize | - | - | - |
| LLaVA | - | - | - |

---

### 2.5 Others

#### 2.5.1 Domain-specific Applications

##### Autonomous driving
| Title | Conference | Year | Code |
|-------|------------|------|------|
| ULTRAFEEDBACK | - | - | - |
| HelpSteer | - | - | - |
| LEMA | - | - | - |

##### Finance
| Title | Conference | Year | Code |
|-------|------------|------|------|
| ULTRAFEEDBACK | - | - | - |
| HelpSteer | - | - | - |
| LEMA | - | - | - |

##### Medical
| Title | Conference | Year | Code |
|-------|------------|------|------|
| ULTRAFEEDBACK | - | - | - |
| HelpSteer | - | - | - |
| LEMA | - | - | - |

##### Law
| Title | Conference | Year | Code |
|-------|------------|------|------|
| ULTRAFEEDBACK | - | - | - |
| HelpSteer | - | - | - |
| LEMA | - | - | - |

##### Education
| Title | Conference | Year | Code |
|-------|------------|------|------|
| ULTRAFEEDBACK | - | - | - |
| HelpSteer | - | - | - |
| LEMA | - | - | - |

---

#### 2.5.2 Structure-specific Applications

##### Time series
| Title | Conference | Year | Code |
|-------|------------|------|------|
| BAD | - | - | - |
| BEAVERTAILS | - | - | - |
| PRM800K | - | - | - |
| WebGPT | - | - | - |

##### Tabular
| Title | Conference | Year | Code |
|-------|------------|------|------|
| BAD | - | - | - |
| BEAVERTAILS | - | - | - |
| PRM800K | - | - | - |
| WebGPT | - | - | - |

##### Graph
| Title | Conference | Year | Code |
|-------|------------|------|------|
| BAD | - | - | - |
| BEAVERTAILS | - | - | - |
| PRM800K | - | - | - |
| WebGPT | - | - | - |

---

## 3. Challenges & Future Directions

### 3.1 Model Collapse
| Title | Conference | Year | Code |
|-------|------------|------|------|
| d-RLAIF | - | - | - |
| LLM2LLM | - | - | - |
| Wizardmath | - | - | - |
| STaR | - | - | - |
| SciGLM | - | - | - |
| ChemLLM | - | - | - |

---

### 3.2 Active Synthesis
| Title | Conference | Year | Code |
|-------|------------|------|------|
| LLMs4Synthesis | - | - | - |
| CoRAL | - | - | - |
| FORD | - | - | - |
| LTGC | - | - | - |

---

### 3.3 Synthetic Data Evaluation
| Title | Conference | Year | Code |
|-------|------------|------|------|
| DataDreamer | - | - | - |
| HARMONIC | - | - | - |

---

### 3.4 Multi-Modal Synthesis
| Title | Conference | Year | Code |
|-------|------------|------|------|
| PANDA | - | - | - |
| REGA | - | - | - |

---

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
