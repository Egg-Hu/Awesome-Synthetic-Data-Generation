<h1 align="center">Awesome Synthetic Data Generation</h1>

<div align="center">

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
![Forks](https://img.shields.io/github/forks/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
<a href='https://arxiv.org/pdf/2409.18169'><img src='https://img.shields.io/badge/arXiv-2409.18169-b31b1b.svg'></a>

</div>

<p align="center">
    <b>A Comprehensive Survey and Curated Collection of Resources on Synthetic Data Generation.</b>
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
  - [2.1 Data-Centric AI](#21-data-centric-ai)
    - [Data Accessibility](#data-accessibility)
    - [Data Refinement](#data-refinement)
  - [2.2 Model-Centric AI](#22-model-centric-ai)
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

**Synthesis From Scratch**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute Zero: Reinforced Self-Play Reasoning With Zero Data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [Condor: Enhance LLM Alignment With Knowledge-Driven Data Synthesis And Refinement](https://arxiv.org/abs/2501.12273) | 2025 | arXiv |
| [DreamTeacher: Pretraining Image Backbones With Deep Generative Models](https://arxiv.org/abs/2307.07487) | 2023 | ICCV |
| [Learning Vision From Models Rivals Learning Vision From Data](https://arxiv.org/abs/2312.17742) | 2024 | CVPR |
| [MathVista: Evaluating Mathematical Reasoning Of Foundation Models In Visual Contexts](https://arxiv.org/abs/2310.02255) | 2023 | arXiv |
| [Self-Instruct: Aligning Language Models With Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | 2022 | arXiv |
| [Synthesizing Post-Training Data For LLMs Through Multi-Agent Simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |
| [Training Verifiers To Solve Math Word Problems](https://arxiv.org/abs/2110.14168) | 2021 | arXiv |

**Synthesis From Seeds**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [CREAM: Consistency Regularized Self-Rewarding Language Models](https://arxiv.org/abs/2410.12735) | 2024 | arXiv |
| [Distribution-Aware Data Expansion With Diffusion Models](http://papers.nips.cc/paper_files/paper/2024/hash/ba2e53000899e45e6018f639cb7469fa-Abstract-Conference.html) | 2024 | NeurIPS |
| [Expanding Small-Scale Datasets With Guided Imagination](http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html) | 2023 | NeurIPS |
| [Human-Guided Image Generation For Expanding Small-Scale Training Image Datasets](https://doi.org/10.1109/TVCG.2025.3567053) | 2025 | IEEE TVCG |
| [Mitigating Catastrophic Forgetting In Large Language Models With Self-Synthesized Rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |
| [Real-Fake: Effective Training Data Synthesis Through Distribution Matching](https://arxiv.org/abs/2310.10402) | 2023 | arXiv |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Spread Preference Annotation: Direct Preference Judgment For Efficient LLM Alignment](https://arxiv.org/abs/2406.04412) | 2025 | ICLR |
| [Towards Automating Text Annotation: A Case Study On Semantic Proximity Annotation Using GPT-4](https://arxiv.org/abs/2407.04130) | 2024 | arXiv |
| [Training On Thin Air: Improve Image Classification With Generated Data](https://arxiv.org/abs/2305.15316) | 2023 | arXiv |

**Synthesis From Structure**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Graph-Based Synthetic Data Pipeline For Scaling High-Quality Reasoning Instructions](https://arxiv.org/abs/2412.08864) | 2024 | arXiv |
| [ControlMath: Controllable Data Generation Promotes Math Generalist Models](https://doi.org/10.18653/v1/2024.emnlp-main.680) | 2024 | EMNLP |
| [Distill Visual Chart Reasoning Ability From LLMs To MLLMs](https://doi.org/10.48550/arXiv.2410.18798) | 2025 | arXiv |
| [Enhancing Logical Reasoning In Large Language Models Through Graph-Based Synthetic Data](https://arxiv.org/abs/2409.12437) | 2024 | arXiv |
| [InfinityMath: A Scalable Instruction Tuning Dataset In Programmatic Mathematical Reasoning](https://doi.org/10.1145/3627673.3679122) | 2024 | ACM |
| [LogicTree: Improving Complex Reasoning Of LLMs Via Instantiated Multi-Step Synthetic Logical Data](https://openreview.net/pdf?id=z4AMrCOetn) | 2025 | NeurIPS |
| [MathScale: Scaling Instruction Tuning For Mathematical Reasoning](https://arxiv.org/abs/2403.02884) | 2024 | ICML |
| [Synthesize-On-Graph: Knowledgeable Synthetic Data Generation For Continued Pre-Training Of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Synthetic Data (Almost) From Scratch: Generalized Instruction Tuning For Language Models](https://arxiv.org/abs/2402.13064) | 2024 | arXiv |

**Synthesis With Evolution**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Survey On LLM-As-A-Judge](https://arxiv.org/abs/2411.15594) | 2024 | arXiv |
| [Beyond Human Data: Scaling Self-Training For Problem-Solving With Language Models](https://arxiv.org/abs/2312.06585) | 2023 | arXiv |
| [CREAM: Consistency Regularized Self-Rewarding Language Models](https://arxiv.org/abs/2410.12735) | 2024 | arXiv |
| [DualFormer: Controllable Fast And Slow Thinking By Learning With Randomized Reasoning Traces](https://arxiv.org/abs/2410.09918) | 2024 | arXiv |
| [Enhancing LLM Reasoning With Iterative DPO: A Comprehensive Empirical Investigation](https://arxiv.org/abs/2503.12854) | 2025 | arXiv |
| [Feedback-Guided Data Synthesis For Imbalanced Classification](https://arxiv.org/abs/2310.00158) | 2023 | arXiv |
| [Improving CLIP Training With Language Rewrites](http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html) | 2023 | NeurIPS |
| [R-Zero: Self-Evolving Reasoning LLM From Zero Data](https://arxiv.org/abs/2508.05004) | 2025 | arXiv |
| [RAFT: Reward Ranked Fine-Tuning For Generative Foundation Model Alignment](https://arxiv.org/abs/2304.06767) | 2023 | arXiv |
| [ReFT: Reasoning With Reinforced Fine-Tuning](https://doi.org/10.18653/v1/2024.acl-long.410) | 2024 | ACL |
| [Scaling Laws Of Synthetic Images For Model Training... For Now](https://doi.org/10.1109/CVPR52733.2024.00705) | 2024 | CVPR |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Spend Wisely: Maximizing Post-Training Gains In Iterative Synthetic Data Bootstrapping](https://arxiv.org/abs/2501.18962) | 2025 | arXiv |
| [Spread Preference Annotation: Direct Preference Judgment For Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [STAR: Bootstrapping Reasoning With Reasoning](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | 2022 | NeurIPS |
| [SwS: Self-Aware Weakness-Driven Problem Synthesis In Reinforcement Learning For LLM Reasoning](https://arxiv.org/abs/2506.08989) | 2025 | arXiv |
| [Synthetic Data From Diffusion Models Improves ImageNet Classification](https://arxiv.org/abs/2304.08466) | 2023 | arXiv |
| [Unleashing Reasoning Capability Of LLMs Via Scalable Question Synthesis From Scratch](https://arxiv.org/abs/2410.18693) | 2024 | arXiv |

### 1.2 Inversion-Based Synthesis

**Data-Space Inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Dreaming To Distill: Data-Free Knowledge Transfer Via DeepInversion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) | 2020 | CVPR |
| [Reverse-Engineered Reasoning For Open-Ended Generation](https://arxiv.org/abs/2509.06160) | 2025 | arXiv |

**Latent-Space Inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Contrastive Model Inversion For Data-Free Knowledge Distillation](https://arxiv.org/abs/2105.08584) | 2021 | arXiv |
| [Deep Image Prior](https://arxiv.org/abs/1711.10925) | 2018 | CVPR |
| [Generative Model Inversion Through The Lens Of The Manifold Hypothesis](https://arxiv.org/abs/2509.20177) | 2025 | arXiv |
| [Knowledge-Enriched Distributional Model Inversion Attacks](https://arxiv.org/abs/2010.04092) | 2021 | ICCV |
| [Label-Only Model Inversion Attacks Via Boundary Repulsion](https://arxiv.org/abs/2203.01925) | 2022 | CVPR |
| [Label-Only Model Inversion Attacks Via Knowledge Transfer](http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html) | 2023 | NeurIPS |
| [Learning To Learn From APIs: Black-Box Data-Free Meta-Learning](https://arxiv.org/abs/2305.18413) | 2023 | arXiv |
| [MIRROR: Model Inversion For Deep Learning Network With High Fidelity](https://par.nsf.gov/servlets/purl/10376663) | 2022 | NDSS |
| [Open-Vocabulary Customization From CLIP Via Data-Free Knowledge Distillation](https://openreview.net/forum?id=1aF2D2CPHi) | 2025 | ICLR |
| [Plug & Play Attacks: Towards Robust And Flexible Model Inversion Attacks](https://arxiv.org/abs/2201.12179) | 2022 | arXiv |
| [Pseudo-Private Data Guided Model Inversion Attacks](http://papers.nips.cc/paper_files/paper/2024/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) | 2024 | NeurIPS |
| [Re-Thinking Model Inversion Attacks Against Deep Neural Networks](https://doi.org/10.1109/CVPR52729.2023.01572) | 2023 | CVPR |
| [Reinforcement Learning-Based Black-Box Model Inversion Attacks](https://doi.org/10.1109/CVPR52729.2023.01964) | 2023 | CVPR |
| [The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html) | 2020 | CVPR |
| [Variational Model Inversion Attacks](https://arxiv.org/abs/2201.10787) | 2021 | NeurIPS |

### 1.3 Simulation-Based Synthesis

**Agent-Based Simulation**

| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [AutoGen: Enabling Next-Gen LLM Applications Via Multi-Agent Conversations](https://openreview.net/forum?id=BAakY1hNKS) | 2024 | COLM |
| [CAMEL: Communicative Agents For "Mind" Exploration Of Large Language Model Society](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html) | 2023 | NeurIPS |
| [MetaGPT: Meta Programming For A Multi-Agent Collaborative Framework](https://openreview.net/forum?id=VtmBAGCN7o) | 2023 | ICLR |
| [Synthesizing Post-Training Data For LLMs Through Multi-Agent Simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |

**Platform-Based Simulation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [AutoGen: Enabling Next-Gen LLM Applications Via Multi-Agent Conversations](https://openreview.net/forum?id=BAakY1hNKS) | 2024 | COLM |
| [CAMEL: Communicative Agents For "Mind" Exploration Of Large Language Model Society](http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html) | 2023 | NeurIPS |
| [MetaGPT: Meta Programming For A Multi-Agent Collaborative Framework](https://openreview.net/forum?id=VtmBAGCN7o) | 2023 | ICLR |
| [Omniverse Replicator: Synthetic Data Generation For AI](https://scholar.google.com/scholar?q=Omniverse%20Replicator%3A%20Synthetic%20Data%20Generation%20for%20AI) | 2021 | arXiv |
| [Synthesizing Post-Training Data For LLMs Through Multi-Agent Simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |
| [Unity Perception: Generate Synthetic Data For Computer Vision](https://arxiv.org/abs/2107.04259) | 2022 | arXiv |

### 1.4 Augmentation-Based Synthesis

**Rule-Based Augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [CutMix: Regularization Strategy To Train Strong Classifiers With Localizable Features](https://doi.org/10.1109/ICCV.2019.00612) | 2019 | ICCV |
| [EDA: Easy Data Augmentation Techniques For Boosting Performance On Text Classification Tasks](https://arxiv.org/abs/1901.11196) | 2019 | arXiv |
| [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | 2017 | arXiv |

**Generative Augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [DART-Math: Difficulty-Aware Rejection Tuning For Mathematical Problem-Solving](http://papers.nips.cc/paper_files/paper/2024/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html) | 2024 | NeurIPS |
| [Data Augmentation For Image Classification Using Generative AI](https://doi.org/10.1109/WACV61041.2025.00410) | 2025 | WACV |
| [DetDiffusion: Synergizing Generative And Perceptive Models For Enhanced Data Generation And Perception](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DetDiffusion_Synergizing_Generative_and_Perceptive_Models_for_Enhanced_Data_Generation_CVPR_2024_paper.html) | 2024 | CVPR |
| [DiffuseMix: Label-Preserving Data Augmentation With Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Islam_DiffuseMix_Label-Preserving_Data_Augmentation_with_Diffusion_Models_CVPR_2024_paper.pdf) | 2024 | CVPR |
| [DreamDA: Generative Data Augmentation With Diffusion Models](https://arxiv.org/abs/2403.12803) | 2024 | arXiv |
| [Learning To Augment Synthetic Images For Sim2Real Policy Transfer](https://arxiv.org/abs/1903.07740) | 2019 | IROS |
| [On The Diversity Of Synthetic Data And Its Impact On Training Large Language Models](https://arxiv.org/abs/2410.15226) | 2024 | arXiv |
| [Scaling Synthetic Data Creation With 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094) | 2024 | arXiv |
| [Self-Improving Diffusion Models With Synthetic Data](https://arxiv.org/abs/2408.16333) | 2024 | arXiv |
| [Synthetic Continued Pretraining](https://arxiv.org/abs/2409.07431) | 2024 | arXiv |
---

## 2. Applications

### 2.1 Data-Centric AI

#### Data Accessibility

**Zero/Few-Shot Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute Zero: Reinforced Self-Play Reasoning With Zero Data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [DataDream: Few-Shot Guided Dataset Generation](https://arxiv.org/abs/2407.10910) | 2024 | ECCV |
| [Generating Synthetic Datasets For Few-Shot Prompt Tuning](https://arxiv.org/abs/2410.10865) | 2024 | arXiv |
| [Learning Vision From Models Rivals Learning Vision From Data](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Learning_Vision_from_Models_Rivals_Learning_Vision_from_Data_CVPR_2024_paper.html) | 2024 | CVPR |
| [Prompting-Based Synthetic Data Generation For Few-Shot Question Answering](https://arxiv.org/abs/2405.09335) | 2024 | arXiv |
| [Tuning Language Models As Training Data Generators For Augmentation-Enhanced Few-Shot Learning](https://proceedings.mlr.press/v202/meng23b.html) | 2023 | PMLR |
| [View-Invariant Policy Learning Via Zero-Shot Novel View Synthesis](https://arxiv.org/abs/2409.03685) | 2024 | arXiv |

**Federated Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Data-Free Federated Class Incremental Learning With Diffusion-Based Generative Memory](https://arxiv.org/abs/2405.17457) | 2024 | arXiv |
| [Parameterized Data-Free Knowledge Distillation For Heterogeneous Federated Learning](https://doi.org/10.1016/j.knosys.2025.113502) | 2021 | Knowledge-Based Systems |
| [DENSE: Data-Free One-Shot Federated Learning](http://papers.nips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract-Conference.html) | 2022 | NeurIPS |
| [Federated Knowledge Recycling: Privacy-Preserving Synthetic Data Sharing](https://doi.org/10.1016/j.patrec.2025.02.030) | 2025 | Pattern Recognition Letters |
| [Fine-Tuning Global Model Via Data-Free Knowledge Distillation For Non-IID Federated Learning](https://doi.org/10.1109/CVPR52688.2022.00993) | 2022 | CVPR |
| [Gradient Inversion Of Federated Diffusion Models](https://arxiv.org/abs/2405.20380) | 2024 | arXiv |
| [Synthetic Data Aided Federated Learning Using Foundation Models](https://doi.org/10.48550/arXiv.2407.05174) | 2024 | arXiv |
| [Understanding Data Reconstruction Leakage In Federated Learning From A Theoretical Perspective](https://arxiv.org/abs/2408.12119) | 2024 | arXiv |
| [VertiMRF: Differentially Private Vertical Federated Data Synthesis](https://doi.org/10.1145/3637528.3671771) | 2024 | KDD |

**Data-Free Knowledge Distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Aligning Teacher With Student Preferences For Tailored Training Data Generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [Contrastive Model Inversion For Data-Free Knowledge Distillation](https://arxiv.org/abs/2105.08584) | 2021 | arXiv |
| [Data-Free Knowledge Distillation For Object Detection](https://doi.org/10.1109/WACV48630.2021.00333) | 2021 | WACV |
| [Data-Free Knowledge Distillation Via Feature Exchange And Activation Region Constraint](https://doi.org/10.1109/CVPR52729.2023.02324) | 2023 | CVPR |
| [Data-Free Knowledge Distillation With Soft Targeted Transfer Set Synthesis](https://doi.org/10.1609/aaai.v35i11.17228) | 2021 | AAAI |
| [Data-Free Learning Of Student Networks](https://doi.org/10.1109/ICCV.2019.00361) | 2019 | ICCV |
| [Dreaming To Distill: Data-Free Knowledge Transfer Via DeepInversion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) | 2020 | CVPR |
| [Model Conversion Via Differentially Private Data-Free Distillation](https://arxiv.org/abs/2304.12528) | 2023 | arXiv |
| [Momentum Adversarial Distillation: Handling Large Distribution Shifts In Data-Free Knowledge Distillation](http://papers.nips.cc/paper_files/paper/2022/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html) | 2022 | NeurIPS |
| [Open-Vocabulary Customization From CLIP Via Data-Free Knowledge Distillation](https://openreview.net/forum?id=1aF2D2CPHi) | 2025 | ICLR |
| [Synthetic Image Learning: Preserving Performance And Preventing Membership Inference Attacks](https://doi.org/10.48550/arXiv.2407.15526) | 2025 | Pattern Recognition Letters |
| [Up To 100x Faster Data-Free Knowledge Distillation](https://doi.org/10.1609/aaai.v36i6.20613) | 2022 | AAAI |

**Data-Free Pruning/Quantization**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Distilled Pruning: Using Synthetic Data To Win The Lottery](https://arxiv.org/abs/2307.03364) | 2023 | arXiv |
| [Sharpness-Aware Data Generation For Zero-Shot Quantization](https://doi.org/10.48550/arXiv.2510.07018) | 2024 | arXiv |

**Data-Free Meta-Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Architecture, Dataset And Model-Scale Agnostic Data-Free Meta-Learning](https://doi.org/10.1109/CVPR52729.2023.00747) | 2023 | CVPR |
| [FREE: Faster And Better Data-Free Meta-Learning](https://doi.org/10.1109/CVPR52733.2024.02196) | 2024 | CVPR |
| [Learning To Learn From APIs: Black-Box Data-Free Meta-Learning](https://arxiv.org/abs/2305.18413) | 2023 | arXiv |
| [Meta-Learning Without Data Via Unconditional Diffusion Models](https://doi.org/10.1109/TCSVT.2024.3424572) | 2024 | IEEE Transactions On Circuits And Systems For Video Technology |
| [Task Groupings Regularization: Data-Free Meta-Learning With Heterogeneous Pre-Trained Models](https://arxiv.org/abs/2405.16560) | 2024 | arXiv |
| [Task-Distributionally Robust Data-Free Meta-Learning](https://doi.org/10.48550/arXiv.2311.14756) | 2025 | IEEE Transactions On Pattern Analysis And Machine Intelligence |

**Data-Free Continual Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Data-Free Approach To Mitigate Catastrophic Forgetting In Federated Class Incremental Learning For Vision Tasks](http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html) | 2023 | NeurIPS |
| [Data-Free Federated Class Incremental Learning With Diffusion-Based Generative Memory](https://arxiv.org/abs/2405.17457) | 2024 | arXiv |
| [DDGR: Continual Learning With Deep Diffusion-Based Generative Replay](https://proceedings.mlr.press/v202/gao23e.html) | 2023 | ICML |
| [Mitigating Catastrophic Forgetting In Large Language Models With Self-Synthesized Rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |
| [Self-Distillation Bridges Distribution Gap In Language Model Fine-Tuning](https://arxiv.org/abs/2402.13669) | 2024 | arXiv |

#### Data Refinement

**Dataset Distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [CAFE: Learning To Condense Dataset By Aligning Features](https://doi.org/10.1109/CVPR52688.2022.01188) | 2022 | CVPR |
| [Dataset Condensation With Distribution Matching](https://proceedings.mlr.press/v216/zheng23a.html) | 2023 | ICML |
| [Dataset Condensation With Gradient Matching](https://arxiv.org/abs/2006.05929) | 2020 | arXiv |
| [Dataset Distillation](https://arxiv.org/abs/1811.10959) | 2018 | arXiv |
| [Dataset Distillation By Matching Training Trajectories](https://doi.org/10.1109/CVPRW56347.2022.00521) | 2022 | CVPR |
| [Dataset Distillation Using Neural Feature Regression](http://papers.nips.cc/paper_files/paper/2022/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html) | 2022 | NeurIPS |
| [Dataset-Distillation Generative Model For Speech Emotion Recognition](https://arxiv.org/abs/2406.02963) | 2024 | arXiv |
| [DIM: Distilling Dataset Into Generative Model](https://doi.org/10.1007/978-3-031-93806-1_4) | 2024 | ECCV |
| [Generalizing Dataset Distillation Via Deep Generative Prior](https://doi.org/10.1109/CVPR52729.2023.00364) | 2023 | CVPR |
| [Generative Dataset Distillation Based On Diffusion Model](https://doi.org/10.48550/arXiv.2505.19469) | 2024 | ECCV |
| [Unlocking Dataset Distillation With Diffusion Models](https://openreview.net/forum?id=c6O18DyBBx) | 2025 | NeurIPS |

**Dataset Purification**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [ADBM: Adversarial Diffusion Bridge Model For Reliable Adversarial Purification](https://arxiv.org/abs/2408.00315) | 2024 | arXiv |
| [Better Synthetic Data By Retrieving And Transforming Existing Datasets](https://arxiv.org/abs/2404.14361) | 2024 | arXiv |
| [DataElixir: Purifying Poisoned Dataset To Mitigate Backdoor Attacks Via Diffusion Models](https://doi.org/10.1609/aaai.v38i19.30186) | 2024 | AAAI |
| [Diffusion Models For Adversarial Purification](https://arxiv.org/abs/2205.07460) | 2022 | arXiv |
---

### 2.2 Model-Centric AI

#### General Model Enhancement

**General Ability**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Enhancing Multilingual Language Model With Massive Multilingual Knowledge Triples](https://arxiv.org/abs/2111.10962) | 2021 | arXiv |
| [Fine-Tuning Or Retrieval? Comparing Knowledge Injection In LLMs](https://arxiv.org/abs/2312.05934) | 2023 | arXiv |
| [Large Language Models, Physics-Based Modeling, Experimental Measurements: The Trinity Of Data-Scarce Learning Of Polymer Properties](https://arxiv.org/abs/2407.02770) | 2024 | arXiv |
| [Llemma: An Open Language Model For Mathematics](https://arxiv.org/abs/2310.10631) | 2023 | arXiv |
| [Nemotron-CC: Transforming Common Crawl Into A Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) | 2024 | arXiv |
| [Rephrasing The Web: A Recipe For Compute And Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380) | 2024 | arXiv |
| [SciLitLLM: How To Adapt LLMs For Scientific Literature Understanding](https://arxiv.org/abs/2408.15545) | 2024 | arXiv |
| [Synthesize-On-Graph: Knowledgeable Synthetic Data Generation For Continued Pre-Training Of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Synthetic Continued Pretraining](https://arxiv.org/abs/2409.07431) | 2024 | arXiv |
| [Textbooks Are All You Need](https://doi.org/10.48550/arXiv.2306.11644) | 2023 | arXiv |
| [Textbooks Are All You Need II: Phi-1.5 Technical Report](https://arxiv.org/abs/2309.05463) | 2023 | arXiv |
| [TinyStories: How Small Can Language Models Be And Still Speak Coherent English?](https://doi.org/10.48550/arXiv.2305.07759) | 2023 | arXiv |
| [VILA$^2$: VILA Augmented VILA](https://arxiv.org/abs/2407.17453) | 2024 | arXiv |

#### Domain Model Enhancement

**Reasoning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Graph-Based Synthetic Data Pipeline For Scaling High-Quality Reasoning Instructions](https://arxiv.org/abs/2412.08864) | 2024 | arXiv |
| [Absolute Zero: Reinforced Self-Play Reasoning With Zero Data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [Aligning Teacher With Student Preferences For Tailored Training Data Generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [AutoCoder: Enhancing Code Large Language Model With AIEV-Instruct](https://arxiv.org/abs/2405.14906) | 2024 | arXiv |
| [Boosting Reward Model With Preference-Conditional Multi-Aspect Synthetic Data Generation](https://arxiv.org/abs/2407.16008) | 2024 | arXiv |
| [ControlMath: Controllable Data Generation Promotes Math Generalist Models](https://doi.org/10.18653/v1/2024.emnlp-main.680) | 2024 | EMNLP |
| [From The Least To The Most: Building A Plug-And-Play Visual Reasoner Via Data Synthesis](https://doi.org/10.18653/v1/2024.emnlp-main.284) | 2024 | EMNLP |
| [HexaCoder: Secure Code Generation Via Oracle-Guided Synthetic Training Data](https://arxiv.org/abs/2409.06446) | 2024 | arXiv |
| [HS-STAR: Hierarchical Sampling For Self-Taught Reasoners Via Difficulty Estimation And Budget Reallocation](https://arxiv.org/abs/2505.19866) | 2025 | arXiv |
| [InfinityMath: A Scalable Instruction Tuning Dataset In Programmatic Mathematical Reasoning](https://doi.org/10.1145/3627673.3679122) | 2024 | ACM |
| [Jiuzhang 3.0: Efficiently Improving Mathematical Reasoning By Training Small Data Synthesis Models](http://papers.nips.cc/paper_files/paper/2024/hash/0356216f73660e15670510f5e42b5fa6-Abstract-Conference.html) | 2024 | NeurIPS |
| [Lamini-LM: A Diverse Herd Of Distilled Models From Large-Scale Instructions](https://arxiv.org/abs/2304.14402) | 2023 | arXiv |
| [Learning To Pose Problems: Reasoning-Driven And Solver-Adaptive Data Synthesis For Large Reasoning Models](https://arxiv.org/abs/2511.09907) | 2025 | arXiv |
| [LogicTree: Improving Complex Reasoning Of LLMs Via Instantiated Multi-Step Synthetic Logical Data](https://openreview.net/pdf?id=z4AMrCOetn) | 2025 | NeurIPS |
| [MAmmoTH: Building Math Generalist Models Through Hybrid Instruction Tuning](https://openreview.net/forum?id=yLClGs770I) | 2024 | ICLR |
| [Marco-O1: Towards Open Reasoning Models For Open-Ended Solutions](https://arxiv.org/abs/2411.14405) | 2024 | arXiv |
| [MathCoder: Seamless Code Integration In LLMs For Enhanced Mathematical Reasoning](https://arxiv.org/abs/2310.03731) | 2023 | arXiv |
| [MathScale: Scaling Instruction Tuning For Mathematical Reasoning](https://openreview.net/forum?id=Kjww7ZN47M) | 2024 | ICML |
| [MetaMath: Bootstrap Your Own Mathematical Questions For Large Language Models](https://openreview.net/forum?id=N8N0hgNDRt) | 2024 | ICLR |
| [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](http://papers.nips.cc/paper_files/paper/2024/hash/3d5aa9a7ce28cdc710fbd044fd3610f3-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | NeurIPS |
| [Prismatic Synthesis: Gradient-Based Data Diversification Boosts Generalization In LLM Reasoning](https://arxiv.org/abs/2505.20161) | 2025 | arXiv |
| [Quiet-STaR: Language Models Can Teach Themselves To Think Before Speaking](https://doi.org/10.48550/arXiv.2403.09629) | 2024 | COLM |
| [Refined Direct Preference Optimization With Synthetic Data For Behavioral Alignment Of LLMs](https://arxiv.org/abs/2402.08005) | 2024 | arXiv |
| [Reflection-Tuning: Recycling Data For Better Instruction-Tuning](https://openreview.net/forum?id=xaqoZZqkPU&utm_source=ainews&utm_medium=email&utm_campaign=ainews-reflection-70b-by-matt-from-it-department) | 2023 | NeurIPS |
| [RL On Incorrect Synthetic Data Scales The Efficiency Of LLM Math Reasoning By Eight-Fold](http://papers.nips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html) | 2024 | NeurIPS |
| [Seed-Coder: Let The Code Model Curate Data For Itself](https://arxiv.org/abs/2506.03524) | 2025 | arXiv |
| [Self-Consistency Preference Optimization](https://arxiv.org/abs/2411.04109) | 2024 | arXiv |
| [Self-Play With Execution Feedback: Improving Instruction-Following Capabilities Of Large Language Models](https://arxiv.org/abs/2406.13542) | 2024 | arXiv |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Small Language Models Need Strong Verifiers To Self-Correct Reasoning](https://doi.org/10.18653/v1/2024.findings-acl.924) | 2024 | ACL |
| [Spread Preference Annotation: Direct Preference Judgment For Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [STAR: Bootstrapping Reasoning With Reasoning](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | 2022 | NeurIPS |
| [Strengthening Multimodal Large Language Model With Bootstrapped Preference Optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [Synthesize-On-Graph: Knowledgeable Synthetic Data Generation For Continued Pre-Training Of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Thinking LLMs: General Instruction Following With Thought Generation](https://doi.org/10.48550/arXiv.2410.10630) | 2025 | ICML |
| [ToRA: A Tool-Integrated Reasoning Agent For Mathematical Problem Solving](https://openreview.net/forum?id=Ep0TtjVoap) | 2024 | ICLR |
| [Tree-Instruct: A Preliminary Study Of The Intrinsic Relationship Between Complexity And Alignment](https://aclanthology.org/2024.lrec-main.1460) | 2024 | LREC-COLING |
| [Unleashing Reasoning Capability Of LLMs Via Scalable Question Synthesis From Scratch](https://doi.org/10.48550/arXiv.2410.18693) | 2025 | arXiv |
| [What Makes Good Data For Alignment? A Comprehensive Study Of Automatic Data Selection In Instruction Tuning](https://arxiv.org/abs/2312.15685) | 2023 | arXiv |
| [WizardCoder: Empowering Code Large Language Models With Evol-Instruct](https://arxiv.org/abs/2306.08568) | 2023 | arXiv |
| [WizardLM: Empowering Large Pre-Trained Language Models To Follow Complex Instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |
| [WizardMath: Empowering Mathematical Reasoning For Large Language Models Via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583) | 2023 | arXiv |

**Code**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute Zero: Reinforced Self-Play Reasoning With Zero Data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [AutoCoder: Enhancing Code Large Language Model With AIEV-Instruct](https://arxiv.org/abs/2405.14906) | 2024 | arXiv |
| [Case2Code: Scalable Synthetic Data For Code Generation](https://aclanthology.org/2025.coling-main.733/) | 2025 | COLING |
| [CodeCLM: Aligning Language Models With Tailored Synthetic Data](https://arxiv.org/abs/2404.05875) | 2024 | arXiv |
| [HexaCoder: Secure Code Generation Via Oracle-Guided Synthetic Training Data](https://arxiv.org/abs/2409.06446) | 2024 | arXiv |
| [Increasing LLM Coding Capabilities Through Diverse Synthetic Coding Tasks](https://arxiv.org/abs/2510.23208) | 2025 | arXiv |
| [Seed-Coder: Let The Code Model Curate Data For Itself](https://arxiv.org/abs/2506.03524) | 2025 | arXiv |
| [WizardCoder: Empowering Code Large Language Models With Evol-Instruct](https://arxiv.org/abs/2306.08568) | 2023 | arXiv |
| [WizardLM: Empowering Large Pre-Trained Language Models To Follow Complex Instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |

**Instruction Following**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [#INSTAG: Instruction Tagging For Analyzing Supervised Fine-Tuning Of Large Language Models](https://arxiv.org/abs/2308.07074) | 2023 | arXiv |
| [AlpaGasus: Training A Better Alpaca With Fewer Data](https://arxiv.org/abs/2307.08701) | 2023 | arXiv |
| [Lamini-LM: A Diverse Herd Of Distilled Models From Large-Scale Instructions](https://arxiv.org/abs/2304.14402) | 2023 | arXiv |
| [LIMA: Less Is More For Alignment](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) | 2023 | NeurIPS |
| [Recursive Introspection: Teaching Language Model Agents How To Self-Improve](http://papers.nips.cc/paper_files/paper/2024/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html) | 2024 | NeurIPS |
| [Reflection-Tuning: Recycling Data For Better Instruction-Tuning](https://openreview.net/forum?id=xaqoZZqkPU&utm_source=ainews&utm_medium=email&utm_campaign=ainews-reflection-70b-by-matt-from-it-department) | 2023 | NeurIPS |
| [Selective Reflection-Tuning: Student-Selected Data Recycling For LLM Instruction-Tuning](https://doi.org/10.18653/v1/2024.findings-acl.958) | 2024 | ACL |
| [Self-Instruct: Aligning Language Models With Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | 2022 | arXiv |
| [Self-Play With Execution Feedback: Improving Instruction-Following Capabilities Of Large Language Models](https://arxiv.org/abs/2406.13542) | 2024 | arXiv |
| [Self-Refine Instruction-Tuning For Aligning Reasoning In Language Models](https://arxiv.org/abs/2405.00402) | 2024 | arXiv |
| [SPAR: Self-Play With Tree-Search Refinement To Improve Instruction-Following In Large Language Models](https://arxiv.org/abs/2412.11605) | 2024 | arXiv |
| [Tree-Instruct: A Preliminary Study Of The Intrinsic Relationship Between Complexity And Alignment](https://aclanthology.org/2024.lrec-main.1460) | 2024 | LREC-COLING |
| [What Makes Good Data For Alignment? A Comprehensive Study Of Automatic Data Selection In Instruction Tuning](https://arxiv.org/abs/2312.15685) | 2023 | arXiv |
| [WizardLM: Empowering Large Pre-Trained Language Models To Follow Complex Instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |

**Preference**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Aligning Teacher With Student Preferences For Tailored Training Data Generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [Boosting Reward Model With Preference-Conditional Multi-Aspect Synthetic Data Generation](https://arxiv.org/abs/2407.16008) | 2024 | arXiv |
| [Course-Correction: Safety Alignment Using Synthetic Preferences](https://arxiv.org/abs/2407.16637) | 2024 | arXiv |
| [Refined Direct Preference Optimization With Synthetic Data For Behavioral Alignment Of LLMs](https://doi.org/10.1007/978-3-031-82481-4_7) | 2024 | International Conference On Machine Learning (Workshop/Associated Volume) |
| [Self-Consistency Preference Optimization](https://arxiv.org/abs/2411.04109) | 2024 | arXiv |
| [Self-Directed Synthetic Dialogues And Revisions Technical Report](https://arxiv.org/abs/2407.18421) | 2024 | arXiv |
| [Spread Preference Annotation: Direct Preference Judgment For Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [Strengthening Multimodal Large Language Model With Bootstrapped Preference Optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |

**Reinforcement Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Kimi K2: Open Agentic Intelligence](https://doi.org/10.48550/arXiv.2507.20534) | 2025 | arXiv |
| [RL On Incorrect Synthetic Data Scales The Efficiency Of LLM Math Reasoning By Eight-Fold](http://papers.nips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html) | 2024 | NeurIPS |
| [S1: Simple Test-Time Scaling](https://doi.org/10.48550/arXiv.2501.19393) | 2025 | arXiv |
| [Synthetic Data Generation & Multi-Step RL For Reasoning & Tool Use](https://doi.org/10.48550/arXiv.2504.04736) | 2025 | arXiv |
| [Synthetic Data RL: Task Definition Is All You Need](https://doi.org/10.48550/arXiv.2505.17063) | 2025 | arXiv |

#### Model Evaluation

**In-Context Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Auto-ICL: In-Context Learning Without Human Supervision](https://doi.org/10.48550/arXiv.2311.09263) | 2024 | arXiv |
| [Automatic Chain Of Thought Prompting In Large Language Models](https://openreview.net/forum?id=5NTt8GFjUHkr) | 2023 | ICLR |
| [Demonstration Augmentation For Zero-Shot In-Context Learning](https://doi.org/10.18653/v1/2024.findings-acl.846) | 2024 | ACL |
| [Embracing Collaboration Over Competition: Condensing Multiple Prompts For Visual In-Context Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Embracing_Collaboration_Over_Competition_Condensing_Multiple_Prompts_for_Visual_In-Context_CVPR_2025_paper.html) | 2025 | CVPR |
| [Generate Rather Than Retrieve: Large Language Models Are Strong Context Generators](https://openreview.net/forum?id=fB0hRu9GZUS) | 2023 | arXiv |
| [Privacy-Preserving In-Context Learning With Differentially Private Few-Shot Generation](https://arxiv.org/abs/2309.11765) | 2023 | arXiv |
| [Recitation-Augmented Language Models](https://openreview.net/forum?id=-cqvvvb-NkI) | 2023 | arXiv |
| [Self-Generated In-Context Learning: Leveraging Auto-Regressive Language Models As A Demonstration Generator](https://doi.org/10.48550/arXiv.2206.08082) | 2022 | arXiv |
| [Self-Harmonized Chain Of Thought](https://doi.org/10.18653/v1/2025.naacl-long.53) | 2025 | NAACL |
| [Self-ICL: Zero-Shot In-Context Learning With Self-Generated Demonstrations](https://doi.org/10.18653/v1/2023.emnlp-main.968) | 2023 | EMNLP |
| [Self-Prompting Large Language Models For Zero-Shot Open-Domain QA](https://doi.org/10.18653/v1/2024.naacl-long.17) | 2024 | NAACL |
| [Synthetic Prompting: Generating Chain-Of-Thought Demonstrations For Large Language Models](https://proceedings.mlr.press/v202/shao23a.html) | 2023 | ICML |

**Synthetic Benchmark**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Achilles' Heel Of Mamba: Essential Difficulties Of The Mamba Architecture Demonstrated By Synthetic Data](https://arxiv.org/abs/2509.17514) | 2025 | arXiv |
| [Can You Rely On Your Model Evaluation? Improving Model Evaluation With Synthetic Test Data](http://papers.nips.cc/paper_files/paper/2023/hash/05fb0f4e645cad23e0ab59d6b9901428-Abstract-Conference.html) | 2023 | NeurIPS |
| [Efficacy Of Synthetic Data As A Benchmark](https://arxiv.org/abs/2409.11968) | 2024 | arXiv |
| [Measuring General Intelligence With Generated Games](https://doi.org/10.48550/arXiv.2505.07215) | 2025 | arXiv |
| [SynBench: Task-Agnostic Benchmarking Of Pretrained Representations Using Synthetic Data](https://doi.org/10.48550/arXiv.2210.02989) | 2022 | NeurIPS |
---

### 2.3 Trustworthy AI

#### Privacy

**Privacy-Preserving Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Differentially Private Synthetic Data Via Foundation Model APIs 1: Images](https://openreview.net/pdf?id=YEhQs8POIo) | 2024 | ICLR |
| [Differentially Private Synthetic Data Via Foundation Model APIs 2: Text](https://arxiv.org/abs/2403.01749) | 2024 | arXiv |
| [Does Training With Synthetic Data Truly Protect Privacy?](https://arxiv.org/abs/2502.12976) | 2025 | arXiv |
| [Enhancing The Utility Of Privacy-Preserving Cancer Classification Using Synthetic Data](https://doi.org/10.1007/978-3-031-77789-9_6) | 2024 | MICCAI |
| [Federated Knowledge Recycling: Privacy-Preserving Synthetic Data Sharing](https://arxiv.org/abs/2407.20830) | 2025 | Pattern Recognition Letters |
| [Generate Synthetic Text Approximating The Private Distribution With Differential Privacy](https://www.ijcai.org/proceedings/2024/735) | 2024 | IJCAI |
| [Privacy-Preserving Instructions For Aligning Large Language Models](https://arxiv.org/abs/2402.13659) | 2024 | arXiv |
| [SETSUBUN: Revisiting Membership Inference Game For Evaluating Synthetic Data Generation](https://doi.org/10.2197/ipsjjip.32.757) | 2024 | Journal Of Information Processing |
| [Synthetic Data Aided Federated Learning Using Foundation Models](https://doi.org/10.48550/arXiv.2407.05174) | 2024 | arXiv |
| [Synthetic Image Learning: Preserving Performance And Preventing Membership Inference Attacks](https://doi.org/10.48550/arXiv.2407.15526) | 2025 | Pattern Recognition Letters |
| [Towards Privacy-Preserving Relational Data Synthesis Via Probabilistic Relational Models](https://doi.org/10.1007/978-3-031-70893-0_13) | 2024 | German Conference On Artificial Intelligence |
| [Using Synthetic Data To Mitigate Unfairness And Preserve Privacy Through Single-Shot Federated Learning](https://doi.org/10.48550/arXiv.2409.09532) | 2024 | arXiv |
| [VertiMRF: Differentially Private Vertical Federated Data Synthesis](https://doi.org/10.1145/3637528.3671771) | 2024 | KDD |

**Model Inversion Attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Methodology For Formalizing Model-Inversion Attacks](https://doi.org/10.1109/CSF.2016.32) | 2016 | IEEE Computer Security Foundations Symposium |
| [Gradient Inversion Of Federated Diffusion Models](https://arxiv.org/abs/2405.20380) | 2024 | arXiv |
| [Model Inversion Attacks That Exploit Confidence Information And Basic Countermeasures](https://doi.org/10.1145/2810103.2813677) | 2015 | ACM SIGSAC |
| [The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html) | 2020 | CVPR |
| [Understanding Data Reconstruction Leakage In Federated Learning From A Theoretical Perspective](https://arxiv.org/abs/2408.12119) | 2024 | arXiv |

#### Safety & Security

**Model Stealing Attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Data-Free Model Extraction](https://doi.org/10.48550/arXiv.2507.16969) | 2021 | CVPR |
| [MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation](https://openaccess.thecvf.com/content/CVPR2021/html/Kariyappa_MAZE_Data-Free_Model_Stealing_Attack_Using_Zeroth-Order_Gradient_Estimation_CVPR_2021_paper.html) | 2021 | CVPR |
| [Unifying Multimodal Large Language Model Capabilities And Modalities Via Model Merging](https://arxiv.org/abs/2505.19892) | 2025 | arXiv |

**Adversarial Defense**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Deceptive Diffusion: Generating Synthetic Adversarial Examples](https://doi.org/10.1007/978-3-031-92366-1_25) | 2025 | International Conference On Scale Space And Variational Methods |
| [Explaining And Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) | 2014 | arXiv |
| [Is Synthetic Data All We Need? Benchmarking The Robustness Of Models Trained With Synthetic Images](https://doi.org/10.1109/CVPRW63382.2024.00257) | 2024 | CVPR |
| [Leaving Reality To Imagination: Robust Classification Via Generated Datasets](https://arxiv.org/abs/2302.02503) | 2023 | arXiv |
| [Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness?](https://arxiv.org/abs/2104.09425) | 2021 | arXiv |
| [TSynD: Targeted Synthetic Data Generation For Enhanced Medical Image Classification: Leveraging Epistemic Uncertainty To Improve Model Performance](https://doi.org/10.1007/978-3-658-47422-5_35) | 2024 | Lecture Notes In Computer Science (Springer) |

**Machine Unlearning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Mitigating Catastrophic Forgetting In Large Language Models With Self-Synthesized Rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |

#### Fairness

**De-Bias Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Discover And Cure: Concept-Aware Mitigation Of Spurious Correlation](https://proceedings.mlr.press/v202/wu23w.html) | 2023 | ICML |
| [FADE: Towards Fairness-Aware Generation For Domain Generalization Via Classifier-Guided Score-Based Diffusion Models](https://arxiv.org/abs/2406.09495) | 2024 | arXiv |
| [Generating Informative Samples For Risk-Averse Fine-Tuning Of Downstream Tasks](https://openreview.net/forum?id=kfB5Ciz2XZ) | 2025 | NeurIPS |
| [Know "No" Better: A Data-Driven Approach For Enhancing Negation Awareness In CLIP](https://arxiv.org/abs/2501.10913) | 2025 | arXiv |
| [Modeling Multi-Task Model Merging As Adaptive Projective Gradient Descent](https://doi.org/10.48550/arXiv.2501.01230) | 2025 | ICML |
| [Refined Direct Preference Optimization With Synthetic Data For Behavioral Alignment Of LLMs](https://doi.org/10.1007/978-3-031-82481-4_7) | 2024 | International Conference On Machine Learning (Workshop/Associated Volume) |
| [Strengthening Multimodal Large Language Model With Bootstrapped Preference Optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [SynthFair: Ensuring Subgroup Fairness In Classification Via Synthetic Data Generation](https://link.springer.com/chapter/10.1007/978-3-031-85856-7_26) | 2024 | World Congress In Computer Science, Computer Engineering & Applied Computing |
| [Using Synthetic Data To Mitigate Unfairness And Preserve Privacy Through Single-Shot Federated Learning](https://doi.org/10.48550/arXiv.2409.09532) | 2024 | arXiv |
| [Virus Infection Attack On LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](https://arxiv.org/abs/2509.23041) | 2025 | arXiv |

**Long-Tail Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [DALK: Dynamic Co-Augmentation Of LLMs And KG To Answer Alzheimer's Disease Questions With Scientific Literature](https://arxiv.org/abs/2405.04819) | 2024 | arXiv |
| [Feedback-Guided Data Synthesis For Imbalanced Classification](https://arxiv.org/abs/2310.00158) | 2023 | arXiv |
| [LTGC: Long-Tail Recognition Via Leveraging LLMs-Driven Generated Content](https://doi.org/10.1109/CVPR52733.2024.01845) | 2024 | CVPR |

#### Interpretability

**Explainable AI**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Debiasing Synthetic Data Generated By Deep Generative Models](http://papers.nips.cc/paper_files/paper/2024/hash/4902603fe8cb095b9ada707a19bd151c-Abstract-Conference.html) | 2024 | NeurIPS |
| [Don't Trust Your Eyes: On The (Un)Reliability Of Feature Visualizations](https://arxiv.org/abs/2306.04719) | 2023 | arXiv |
| [Knowledge-Driven AI-Generated Data For Accurate And Interpretable Breast Ultrasound Diagnoses](https://arxiv.org/abs/2407.16634) | 2024 | arXiv |
| [Multifaceted Feature Visualization: Uncovering The Different Types Of Features Learned By Each Neuron In Deep Neural Networks](https://arxiv.org/abs/1602.03616) | 2016 | arXiv |
| [Understanding Deep Image Representations By Inverting Them](https://doi.org/10.1109/CVPR.2015.7299155) | 2015 | CVPR |

#### Governance

**Data Watermarking**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Can Watermarking Large Language Models Prevent Copyrighted Text Generation And Hide Training Data?](https://doi.org/10.1609/aaai.v39i23.34684) | 2025 | AAAI |
| [TimeWak: Temporal Chained-Hashing Watermark For Time Series Data](https://arxiv.org/abs/2506.06407) | 2025 | arXiv |
---

### 2.4 Embodied AI

#### Perception

**Visual Sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [DexTreme: Transfer Of Agile In-Hand Manipulation From Simulation To Reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |
| [Habitat 3.0: A Co-Habitat For Humans, Avatars, And Robots](https://openreview.net/forum?id=4znwzG92CE) | 2024 | ICLR |
| [Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis Of 3D Scene Scale And Realism Tradeoffs For ObjectGoal Navigation](https://doi.org/10.1109/CVPR52733.2024.01550) | 2024 | CVPR |
| [MuJoCo: A Physics Engine For Model-Based Control](https://doi.org/10.1109/IROS.2012.6386109) | 2012 | IROS |
| [Orbit: A Unified Simulation Framework For Interactive Robot Learning Environments](https://doi.org/10.1109/LRA.2023.3270034) | 2023 | IEEE Robotics And Automation Letters |
| [ProcTHOR: Large-Scale Embodied AI Using Procedural Generation](http://papers.nips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html) | 2022 | NeurIPS |
| [Re$^3$Sim: Generating High-Fidelity Simulation Data Via 3D-Photorealistic Real-To-Sim For Robotic Manipulation](https://arxiv.org/abs/2502.08645) | 2025 | arXiv |
| [SplatSim: Zero-Shot Sim2Real Transfer Of RGB Manipulation Policies Using Gaussian Splatting](https://doi.org/10.1109/ICRA55743.2025.11128339) | 2025 | ICRA |

**Force Sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [ARNOLD: A Benchmark For Language-Grounded Task Learning With Continuous States In Realistic 3D Scenes](https://doi.org/10.1109/ICCV51070.2023.01873) | 2023 | ICCV |
| [DexTreme: Transfer Of Agile In-Hand Manipulation From Simulation To Reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |
| [Humanoid-Gym: Reinforcement Learning For Humanoid Robot With Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695) | 2024 | arXiv |

**Sensor Fusion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [EmbodiedGPT: Vision-Language Pre-Training Via Embodied Chain Of Thought](http://papers.nips.cc/paper_files/paper/2023/hash/4ec43957eda1126ad4887995d05fae3b-Abstract-Conference.html) | 2023 | NeurIPS |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | 2023 | arXiv |
| [RT-2: Vision-Language-Action Models Transfer Web Knowledge To Robotic Control](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | CoRL |
| [SpatialVLM: Endowing Vision-Language Models With Spatial Reasoning Capabilities](https://doi.org/10.1109/CVPR52733.2024.01370) | 2024 | CVPR |

#### Interaction

**Trajectory Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [3D Diffusion Policy: Generalizable Visuomotor Policy Learning Via Simple 3D Representations](https://doi.org/10.15607/RSS.2024.XX.067) | 2024 | RSS |
| [DexMimicGen: Automated Data Generation For Bimanual Dexterous Manipulation Via Imitation Learning](https://doi.org/10.1109/ICRA55743.2025.11127809) | 2025 | ICRA |
| [Diffusion Policy: Visuomotor Policy Learning Via Action Diffusion](https://doi.org/10.1177/02783649241273668) | 2023 | RSS |
| [IntervenGen: Interventional Data Generation For Robust And Data-Efficient Robot Imitation Learning](https://doi.org/10.1109/IROS58592.2024.10801523) | 2024 | IROS |
| [MimicGen: A Data Generation System For Scalable Robot Learning Using Human Demonstrations](https://proceedings.mlr.press/v229/mandlekar23a.html) | 2023 | CoRL |
| [ReBot: Scaling Robot Learning With Real-To-Sim-To-Real Robotic Video Synthesis](https://arxiv.org/abs/2503.14526) | 2025 | arXiv |
| [Scaling Robot Learning With Semantically Imagined Experience](https://arxiv.org/abs/2302.11550) | 2023 | arXiv |

**Environment Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [ALFRED: A Benchmark For Interpreting Grounded Instructions For Everyday Tasks](https://openaccess.thecvf.com/content_CVPR_2020/html/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.html) | 2020 | CVPR |
| [HOLODECK: Language Guided Generation Of 3D Embodied AI Environments](https://doi.org/10.1109/CVPR52733.2024.01536) | 2024 | CVPR |
| [LUMINOUS: Indoor Scene Generation For Embodied AI Challenges](https://arxiv.org/abs/2111.05527) | 2021 | arXiv |
| [PARTNR: A Benchmark For Planning And Reasoning In Embodied Multi-Agent Tasks](https://arxiv.org/abs/2411.00081) | 2024 | arXiv |
| [RoboGen: Towards Unleashing Infinite Data For Automated Robot Learning Via Generative Simulation](https://openreview.net/forum?id=SQIDlJd3hN) | 2024 | ICML |
| [TEACH: Task-Driven Embodied Agents That Chat](https://doi.org/10.1609/aaai.v36i2.20097) | 2022 | AAAI |
| [VirtualHome: Simulating Household Activities Via Programs](http://openaccess.thecvf.com/content_cvpr_2018/html/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.html) | 2018 | CVPR |

**Human Behavior Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [3D Human Reconstruction In The Wild With Synthetic Data Using Generative Models](https://arxiv.org/abs/2403.11111) | 2024 | arXiv |
| [Social-LLaVA: Enhancing Robot Navigation Through Human-Language Reasoning In Social Spaces](https://arxiv.org/abs/2501.09024) | 2024 | arXiv |
| [Socially Compliant Navigation Dataset (SCAND): A Large-Scale Dataset Of Demonstrations For Social Navigation](https://doi.org/10.1109/LRA.2022.3184025) | 2022 | IEEE Robotics And Automation Letters |
| [Toward Human-Like Social Robot Navigation: A Large-Scale, Multi-Modal, Social Human Navigation Dataset](https://doi.org/10.1109/IROS55552.2023.10342447) | 2023 | IROS |

#### Generalization

**Cross-Embodiment Training**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945) | 2024 | arXiv |
| [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) | 2024 | arXiv |
| [Open X-Embodiment: Robotic Learning Datasets And RT-X Models: Open X-Embodiment Collaboration 0](https://doi.org/10.48550/arXiv.2310.08864) | 2024 | ICRA |
| [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) | 2024 | arXiv |

**Vision-Language-Action Models**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion](http://papers.nips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html) | 2024 | NeurIPS |
| [Learning Universal Policies Via Text-Guided Video Generation](http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html) | 2023 | NeurIPS |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | 2023 | arXiv |
| [RT-2: Vision-Language-Action Models Transfer Web Knowledge To Robotic Control](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | CoRL |

**Sim-To-Real Transfer**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Bi-Directional Domain Adaptation For Sim2Real Transfer Of Embodied Navigation Agents](https://doi.org/10.1109/LRA.2021.3062303) | 2021 | IEEE Robotics And Automation Letters |
| [DexTreme: Transfer Of Agile In-Hand Manipulation From Simulation To Reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |
| [Re$^3$Sim: Generating High-Fidelity Simulation Data Via 3D-Photorealistic Real-To-Sim For Robotic Manipulation](https://arxiv.org/abs/2502.08645) | 2025 | arXiv |
| [RetinaGAN: An Object-Aware Approach To Sim-To-Real Transfer](https://doi.org/10.1109/ICRA48506.2021.9561157) | 2020 | ICRA |
| [SplatSim: Zero-Shot Sim2Real Transfer Of RGB Manipulation Policies Using Gaussian Splatting](https://doi.org/10.1109/ICRA55743.2025.11128339) | 2025 | ICRA |
| [TransIC: Sim-To-Real Policy Transfer By Learning From Online Correction](https://arxiv.org/abs/2405.10315) | 2024 | arXiv |
---

## 3. Challenges & Future Directions

**Model Collapse**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Theoretical Perspective: How To Prevent Model Collapse In Self-Consuming Training Loops](https://arxiv.org/abs/2502.18865) | 2025 | arXiv |
| [AI Models Collapse When Trained On Recursively Generated Data](https://doi.org/10.48550/arXiv.2410.12954) | 2024 | Nature |
| [Beyond Model Collapse: Scaling Up With Synthesized Data Requires Verification](https://arxiv.org/abs/2406.07515) | 2024 | arXiv |
| [Fairness Feedback Loops: Training On Synthetic Data Amplifies Bias](https://doi.org/10.1145/3630106.3659029) | 2024 | ACM |
| [Is Model Collapse Inevitable? Breaking The Curse Of Recursion By Accumulating Real And Synthetic Data](https://arxiv.org/abs/2404.01413) | 2024 | arXiv |
| [Large Language Models Suffer From Their Own Output: An Analysis Of The Self-Consuming Training Loop](https://doi.org/10.48550/arXiv.2311.16822) | 2023 | arXiv |
| [On The Stability Of Iterative Retraining Of Generative Models On Their Own Data](https://arxiv.org/abs/2310.00429) | 2023 | arXiv |
| [Self-Consuming Generative Models Go Mad](https://openreview.net/forum?id=ShjMHfmPs0) | 2023 | ICLR |
| [Self-Correcting Self-Consuming Loops For Generative Model Training](https://arxiv.org/abs/2402.07087) | 2024 | arXiv |
| [Towards Theoretical Understandings Of Self-Consuming Generative Models](https://arxiv.org/abs/2402.11778) | 2024 | arXiv |

**Active Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Controlled Training Data Generation With Diffusion Models](https://arxiv.org/abs/2403.15309) | 2024 | arXiv |
| [LLM See, LLM Do: Guiding Data Generation To Target Non-Differentiable Objectives](https://arxiv.org/abs/2407.01490) | 2024 | arXiv |

**Synthetic Data Evaluation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A Multi-Faceted Evaluation Framework For Assessing Synthetic Data Generated By Large Language Models](https://arxiv.org/abs/2404.14445) | 2024 | arXiv |

**Multi-Modal Synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Enhanced Visual Instruction Tuning With Synthesized Image-Dialogue Data](https://doi.org/10.18653/v1/2024.findings-acl.864) | 2024 | ACL |
| [Strengthening Multimodal Large Language Model With Bootstrapped Preference Optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [SynthVLM: High-Efficiency And High-Quality Synthetic Data For Vision Language Models](https://doi.org/10.48550/arXiv.2407.20756) | 2024 | CoRL |
---

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back To Top ↑
    </a>
</p>
