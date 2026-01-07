<h1 align="center">Awesome Synthetic Data Generation</h1>

<div align="center">

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)
![Stars](https://img.shields.io/github/stars/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)
![Forks](https://img.shields.io/github/forks/Egg-Hu/Awesome-Synthetic-Data-Generation?style=social)

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
    - [Model Collapse](#model-collapse)
    - [Utility-Privacy Tradeoffs](#utility-privacy-tradeoffs)
    - [Generation-Evaluation Bias](#generation-evaluation-bias)
    - [Active Data Synthesis](#active-data-synthesis)
    - [Synthetic Data Evaluation](#synthetic-data-evaluation)
    - [Multi-Modal Data Synthesis](#multi-modal-data-synthesis)

</details>

---

## 1. Methodologies

### 1.1 Generation-Based Synthesis

**Synthesis from scratch**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute zero: Reinforced self-play reasoning with zero data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [Synthesizing post-training data for llms through multi-agent simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |
| [Learning vision from models rivals learning vision from data](https://arxiv.org/abs/2312.17742) | 2024 | CVPR |
| [Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts](https://arxiv.org/abs/2310.02255) | 2023 | arXiv |
| [Dreamteacher: Pretraining image backbones with deep generative models](https://arxiv.org/abs/2307.07487) | 2023 | ICCV |
| [Self-instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560) | 2022 | arXiv |
| [Training verifiers to solve math word problems](https://arxiv.org/abs/2110.14168) | 2021 | arXiv |

**Synthesis from seeds**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](https://arxiv.org/abs/2406.04412) | 2025 | ICLR |
| [Human-guided image generation for expanding small-scale training image datasets](https://doi.org/10.1109/TVCG.2025.3567053) | 2025 | TVCG |
| [Cream: Consistency regularized self-rewarding language models](https://arxiv.org/abs/2410.12735) | 2024 | arXiv |
| [Distribution-aware data expansion with diffusion models](http://papers.nips.cc/paper_files/paper/2024/hash/ba2e53000899e45e6018f639cb7469fa-Abstract-Conference.html) | 2024 | NeurIPS |
| [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |
| [Towards automating text annotation: A case study on semantic proximity annotation using gpt-4](https://arxiv.org/abs/2407.04130) | 2024 | arXiv |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Real-fake: Effective training data synthesis through distribution matching](https://arxiv.org/abs/2310.10402) | 2023 | arXiv |
| [Expanding small-scale datasets with guided imagination](http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html) | 2023 | NeurIPS |
| [Training on thin air: Improve image classification with generated data](https://arxiv.org/abs/2305.15316) | 2023 | arXiv |

**Synthesis from structure**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Logictree: Improving complex reasoning of LLMs via instantiated multi-step synthetic logical data](https://openreview.net/pdf?id=z4AMrCOetn) | 2025 | NeurIPS |
| [A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Reasoning Instructions](https://arxiv.org/abs/2412.08864) | 2024 | arXiv |
| [MathScale: Scaling Instruction Tuning for Mathematical Reasoning](https://arxiv.org/abs/2403.02884) | 2024 | ICML |
| [ControlMath: Controllable Data Generation Promotes Math Generalist Models](https://doi.org/10.18653/v1/2024.emnlp-main.680) | 2024 | EMNLP |
| [Enhancing logical reasoning in large language models through graph-based synthetic data](https://arxiv.org/abs/2409.12437) | 2024 | arXiv |
| [Infinitymath: A scalable instruction tuning dataset in programmatic mathematical reasoning](https://doi.org/10.1145/3627673.3679122) | 2024 | CIKM |
| [Synthetic data (almost) from scratch: Generalized instruction tuning for language models](https://arxiv.org/abs/2402.13064) | 2024 | arXiv |

**Synthesis with evolution**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004) | 2025 | arXiv |
| [Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2505.20161) | 2025 | arXiv |
| [HS-STAR: Hierarchical Sampling for Self-Taught Reasoners via Difficulty Estimation and Budget Reallocation](https://arxiv.org/abs/2505.19866) | 2025 | arXiv |
| [Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation](https://arxiv.org/abs/2503.12854) | 2025 | arXiv |
| [Spend wisely: Maximizing post-training gains in iterative synthetic data boostrapping](https://arxiv.org/abs/2501.18962) | 2025 | arXiv |
| [Scaling laws of synthetic images for model training... for now](https://doi.org/10.1109/CVPR52733.2024.00705) | 2024 | CVPR |
| [Cream: Consistency regularized self-rewarding language models](https://arxiv.org/abs/2410.12735) | 2024 | arXiv |
| [Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces](https://arxiv.org/abs/2410.09918) | 2024 | arXiv |
| [Improving clip training with language rewrites](http://papers.nips.cc/paper_files/paper/2023/hash/6fa4d985e7c434002fb6289ab9b2d654-Abstract-Conference.html) | 2023 | NeurIPS |
| [Feedback-guided data synthesis for imbalanced classification](https://arxiv.org/abs/2310.00158) | 2023 | arXiv |
| [Raft: Reward ranked finetuning for generative foundation model alignment](https://arxiv.org/abs/2304.06767) | 2023 | arXiv |
| [Synthetic data from diffusion models improves imagenet classification](https://arxiv.org/abs/2304.08466) | 2023 | arXiv |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [Star: Bootstrapping reasoning with reasoning](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | 2022 | NeurIPS |
| [Unleashing reasoning capability of llms via scalable question synthesis from scratch](https://arxiv.org/abs/2410.18693) | 2024 | arXiv |
| [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) | 2024 | arXiv |

### 1.2 Inversion-Based Synthesis

**Data-space inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Reverse-Engineered Reasoning for Open-Ended Generation](https://arxiv.org/abs/2509.06160) | 2025 | arXiv |
| [Dreaming to distill: Data-free knowledge transfer via deepinversion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) | 2020 | CVPR |

**Latent-space inversion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Generative Model Inversion Through the Lens of the Manifold Hypothesis](https://arxiv.org/abs/2509.20177) | 2025 | arXiv |
| [Pseudo-private data guided model inversion attacks](http://papers.nips.cc/paper_files/paper/2024/hash/3a797b10ff20562b1ecee0d4e914c1c7-Abstract-Conference.html) | 2024 | NeurIPS |
| [Open-vocabulary customization from clip via data-free knowledge distillation](https://openreview.net/forum?id=1aF2D2CPHi) | 2025 | ICLR |
| [Reinforcement learning-based black-box model inversion attacks](https://doi.org/10.1109/CVPR52729.2023.01964) | 2023 | CVPR |
| [Re-thinking model inversion attacks against deep neural networks](https://doi.org/10.1109/CVPR52729.2023.01572) | 2023 | CVPR |
| [Learning to Learn from APIs: Black-Box Data-Free Meta-Learning](https://arxiv.org/abs/2305.18413) | 2023 | arXiv |
| [Label-only model inversion attacks via knowledge transfer](http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html) | 2023 | NeurIPS |
| [Label-only model inversion attacks via boundary repulsion](https://arxiv.org/abs/2203.01925) | 2022 | CVPR |
| [Mirror: Model inversion for deep learning network with high fidelity](https://par.nsf.gov/servlets/purl/10376663) | 2022 | NDSS |
| [Variational model inversion attacks](https://arxiv.org/abs/2201.10787) | 2021 | NeurIPS |
| [Knowledge-enriched distributional model inversion attacks](https://arxiv.org/abs/2010.04092) | 2021 | ICCV |
| [Contrastive model inversion for data-free knowledge distillation](https://arxiv.org/abs/2105.08584) | 2021 | arXiv |
| [Deep image prior](https://arxiv.org/abs/1711.10925) | 2018 | CVPR |
| [The secret revealer: Generative model-inversion attacks against deep neural networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html) | 2020 | CVPR |
| [Plug & play attacks: Towards robust and flexible model inversion attacks](https://arxiv.org/abs/2201.12179) | 2022 | arXiv |

### 1.3 Simulation-Based Synthesis

**Agent-based simulation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Synthesizing post-training data for llms through multi-agent simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |
| [Autogen: Enabling next-gen LLM applications via multi-agent conversations](https://openreview.net/forum?id=BAakY1hNKS) | 2024 | COLM |
| [Camel: Communicative agents for" mind" exploration of large language model society](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html) | 2023 | NeurIPS |
| [MetaGPT: Meta programming for a multi-agent collaborative framework](https://openreview.net/forum?id=VtmBAGCN7o) | 2023 | ICLR |

**Platform-based simulation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Synthesizing post-training data for llms through multi-agent simulation](https://arxiv.org/abs/2410.14251) | 2024 | arXiv |
| [Autogen: Enabling next-gen LLM applications via multi-agent conversations](https://openreview.net/forum?id=BAakY1hNKS) | 2024 | COLM |
| [Camel: Communicative agents for" mind" exploration of large language model society](http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html) | 2023 | NeurIPS |
| [MetaGPT: Meta programming for a multi-agent collaborative framework](https://openreview.net/forum?id=VtmBAGCN7o) | 2023 | ICLR |
| [Unity Perception: Generate Synthetic Data for Computer Vision](https://arxiv.org/abs/2107.04259) | 2022 | arXiv |
| [Omniverse Replicator: Synthetic Data Generation for AI](https://scholar.google.com/scholar?q=Omniverse%20Replicator%3A%20Synthetic%20Data%20Generation%20for%20AI) | 2021 | arXiv |

### 1.4 Augmentation-Based Synthesis

**Rule-based augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Cutmix: Regularization strategy to train strong classifiers with localizable features](https://doi.org/10.1109/ICCV.2019.00612) | 2019 | ICCV |
| [EDA: Easy data augmentation techniques for boosting performance on text classification tasks](https://arxiv.org/abs/1901.11196) | 2019 | arXiv |
| [mixup: Beyond empirical risk minimization](https://arxiv.org/abs/1710.09412) | 2017 | arXiv |

**Generative augmentation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Data augmentation for image classification using generative ai](https://doi.org/10.1109/WACV61041.2025.00410) | 2025 | WACV |
| [Self-improving diffusion models with synthetic data](https://arxiv.org/abs/2408.16333) | 2024 | arXiv |
| [On the diversity of synthetic data and its impact on training large language models](https://arxiv.org/abs/2410.15226) | 2024 | arXiv |
| [Scaling synthetic data creation with 1,000,000,000 personas](https://arxiv.org/abs/2406.20094) | 2024 | arXiv |
| [Synthetic continued pretraining](https://arxiv.org/abs/2409.07431) | 2024 | arXiv |
| [Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving](http://papers.nips.cc/paper_files/paper/2024/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html) | 2024 | NeurIPS |
| [Detdiffusion: Synergizing generative and perceptive models for enhanced data generation and perception](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_DetDiffusion_Synergizing_Generative_and_Perceptive_Models_for_Enhanced_Data_Generation_CVPR_2024_paper.html) | 2024 | CVPR |
| [Diffusemix: Label-preserving data augmentation with diffusion models](https://openaccess.thecvf.com/content/CVPR2024/papers/Islam_DiffuseMix_Label-Preserving_Data_Augmentation_with_Diffusion_Models_CVPR_2024_paper.pdf) | 2024 | CVPR |
| [Dreamda: Generative data augmentation with diffusion models](https://arxiv.org/abs/2403.12803) | 2024 | arXiv |
| [Learning to augment synthetic images for sim2real policy transfer](https://arxiv.org/abs/1903.07740) | 2019 | IROS |

---

## 2. Applications

### 2.1 Data-centric AI

#### Data Accessibility

**Zero/Few-shot learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute zero: Reinforced self-play reasoning with zero data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [Generating synthetic datasets for few-shot prompt tuning](https://arxiv.org/abs/2410.10865) | 2024 | arXiv |
| [View-invariant policy learning via zero-shot novel view synthesis](https://arxiv.org/abs/2409.03685) | 2024 | arXiv |
| [Prompting-based synthetic data generation for few-shot question answering](https://arxiv.org/abs/2405.09335) | 2024 | arXiv |
| [Datadream: Few-shot guided dataset generation](https://arxiv.org/abs/2407.10910) | 2024 | ECCV |
| [Learning vision from models rivals learning vision from data](https://openaccess.thecvf.com/content/CVPR2024/html/Tian_Learning_Vision_from_Models_Rivals_Learning_Vision_from_Data_CVPR_2024_paper.html) | 2024 | CVPR |
| [Tuning language models as training data generators for augmentation-enhanced few-shot learning](https://proceedings.mlr.press/v202/meng23b.html) | 2023 | PMLR |

**Federated learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Federated Knowledge Recycling: Privacy-preserving synthetic data sharing](https://doi.org/10.1016/j.patrec.2025.02.030) | 2025 | PATTERN RECOGN LETT |
| [Synthetic data aided federated learning using foundation models](https://doi.org/10.48550/arXiv.2407.05174) | 2024 | arXiv |
| [Understanding Data Reconstruction Leakage in Federated Learning from a Theoretical Perspective](https://arxiv.org/abs/2408.12119) | 2024 | arXiv |
| [Gradient inversion of federated diffusion models](https://arxiv.org/abs/2405.20380) | 2024 | arXiv |
| [VertiMRF: Differentially Private Vertical Federated Data Synthesis](https://doi.org/10.1145/3637528.3671771) | 2024 | KDD |
| [Data-Free Federated Class Incremental Learning with Diffusion-Based Generative Memory](https://arxiv.org/abs/2405.17457) | 2024 | arXiv |
| [Fine-tuning global model via data-free knowledge distillation for non-iid federated learning](https://doi.org/10.1109/CVPR52688.2022.00993) | 2022 | CVPR |
| [DENSE: Data-Free One-Shot Federated Learning](http://papers.nips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract-Conference.html) | 2022 | NeurIPS |
| [Parameterized Data-free knowledge distillation for heterogeneous federated learning](https://doi.org/10.1016/j.knosys.2025.113502) | 2021 | ICML |

**Data-free knowledge distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Open-vocabulary customization from clip via data-free knowledge distillation](https://openreview.net/forum?id=1aF2D2CPHi) | 2025 | ICLR |
| [Synthetic image learning: Preserving performance and preventing membership inference attacks](https://doi.org/10.48550/arXiv.2407.15526) | 2025 | PATTERN RECOGN LETT |
| [Aligning teacher with student preferences for tailored training data generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [Model Conversion via Differentially Private Data-Free Distillation](https://arxiv.org/abs/2304.12528) | 2023 | arXiv |
| [Data-Free Knowledge Distillation via Feature Exchange and Activation Region Constraint](https://doi.org/10.1109/CVPR52729.2023.02324) | 2023 | CVPR |
| [Re-thinking model inversion attacks against deep neural networks](https://doi.org/10.1109/CVPR52729.2023.01572) | 2023 | CVPR |
| [Up to 100x faster data-free knowledge distillation](https://doi.org/10.1609/aaai.v36i6.20613) | 2022 | AAAI |
| [Momentum Adversarial Distillation: Handling Large Distribution Shifts in Data-Free Knowledge Distillation](http://papers.nips.cc/paper_files/paper/2022/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html) | 2022 | NeurIPS |
| [Dreaming to distill: Data-free knowledge transfer via deepinversion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) | 2020 | CVPR |
| [Data-free knowledge distillation with soft targeted transfer set synthesis](https://doi.org/10.1609/aaai.v35i11.17228) | 2021 | AAAI |
| [Contrastive model inversion for data-free knowledge distillation](https://arxiv.org/abs/2105.08584) | 2021 | arXiv |
| [Data-free knowledge distillation for object detection](https://doi.org/10.1109/WACV48630.2021.00333) | 2021 | WACV |
| [Data-free learning of student networks](https://doi.org/10.1109/ICCV.2019.00361) | 2019 | ICCV |

**Data-free pruning/quantization**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Sharpness-aware data generation for zero-shot quantization](https://doi.org/10.48550/arXiv.2510.07018) | 2024 | arXiv |
| [Distilled Pruning: Using Synthetic Data to Win the Lottery](https://arxiv.org/abs/2307.03364) | 2023 | arXiv |

**Data-free meta-learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Task-distributionally robust data-free meta-learning](https://doi.org/10.48550/arXiv.2311.14756) | 2025 | TPAMI |
| [FREE: Faster and Better Data-Free Meta-Learning](https://doi.org/10.1109/CVPR52733.2024.02196) | 2024 | CVPR |
| [Meta-learning without data via unconditional diffusion models](https://doi.org/10.1109/TCSVT.2024.3424572) | 2024 | TCSVT |
| [Task Groupings Regularization: Data-Free Meta-Learning with Heterogeneous Pre-trained Models](https://arxiv.org/abs/2405.16560) | 2024 | arXiv |
| [Architecture, Dataset and Model-Scale Agnostic Data-free Meta-Learning](https://doi.org/10.1109/CVPR52729.2023.00747) | 2023 | CVPR |
| [Learning to Learn from APIs: Black-Box Data-Free Meta-Learning](https://arxiv.org/abs/2305.18413) | 2023 | arXiv |

**Data-free continual learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Data-Free Federated Class Incremental Learning with Diffusion-Based Generative Memory](https://arxiv.org/abs/2405.17457) | 2024 | arXiv |
| [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |
| [Self-distillation bridges distribution gap in language model fine-tuning](https://arxiv.org/abs/2402.13669) | 2024 | arXiv |
| [A data-free approach to mitigate catastrophic forgetting in federated class incremental learning for vision tasks](http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html) | 2023 | NeurIPS |
| [Ddgr: Continual learning with deep diffusion-based generative replay](https://proceedings.mlr.press/v202/gao23e.html) | 2023 | ICML |

#### Data Refinement

**Dataset distillation**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Unlocking Dataset Distillation with Diffusion Models](https://openreview.net/forum?id=c6O18DyBBx) | 2025 | NeurIPS |
| [Generative dataset distillation based on diffusion model](https://doi.org/10.48550/arXiv.2505.19469) | 2024 | ECCV |
| [Dim: Distilling dataset into generative model](https://doi.org/10.1007/978-3-031-93806-1_4) | 2024 | ECCV |
| [Dataset-distillation generative model for speech emotion recognition](https://arxiv.org/abs/2406.02963) | 2024 | arXiv |
| [Generalizing dataset distillation via deep generative prior](https://doi.org/10.1109/CVPR52729.2023.00364) | 2023 | CVPR |
| [Dataset condensation with distribution matching](https://proceedings.mlr.press/v216/zheng23a.html) | 2023 | PMLR |
| [Dataset distillation using neural feature regression](http://papers.nips.cc/paper_files/paper/2022/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html) | 2022 | NeurIPS |
| [Dataset distillation by matching training trajectories](https://doi.org/10.1109/CVPRW56347.2022.00521) | 2022 | CVPR |
| [Cafe: Learning to condense dataset by aligning features](https://doi.org/10.1109/CVPR52688.2022.01188) | 2022 | CVPR |
| [Dataset condensation with gradient matching](https://arxiv.org/abs/2006.05929) | 2020 | arXiv |
| [Dataset distillation](https://arxiv.org/abs/1811.10959) | 2018 | arXiv |

**Dataset purification**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Adbm: Adversarial diffusion bridge model for reliable adversarial purification](https://arxiv.org/abs/2408.00315) | 2024 | arXiv |
| [Better synthetic data by retrieving and transforming existing datasets](https://arxiv.org/abs/2404.14361) | 2024 | arXiv |
| [Dataelixir: Purifying poisoned dataset to mitigate backdoor attacks via diffusion models](https://doi.org/10.1609/aaai.v38i19.30186) | 2024 | AAAI |
| [Diffusion models for adversarial purification](https://arxiv.org/abs/2205.07460) | 2022 | arXiv |
___

### 2.2 Model-centric AI

#### General Model Enhancement

**General ability**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Large language models, physics-based modeling, experimental measurements: the trinity of data-scarce learning of polymer properties](https://arxiv.org/abs/2407.02770) | 2024 | arXiv |
| [Nemotron-CC: Transforming Common Crawl into a refined long-horizon pretraining dataset](https://arxiv.org/abs/2412.02595) | 2024 | arXiv |
| [Rephrasing the web: A recipe for compute and data-efficient language modeling](https://arxiv.org/abs/2401.16380) | 2024 | arXiv |
| [Scilitllm: How to adapt llms for scientific literature understanding](https://arxiv.org/abs/2408.15545) | 2024 | arXiv |
| [Synthetic continued pretraining](https://arxiv.org/abs/2409.07431) | 2024 | arXiv |
| [VILA $^ 2$: VILA Augmented VILA](https://arxiv.org/abs/2407.17453) | 2024 | arXiv |
| [Fine-tuning or retrieval? comparing knowledge injection in llms](https://arxiv.org/abs/2312.05934) | 2023 | arXiv |
| [Llemma: An open language model for mathematics](https://arxiv.org/abs/2310.10631) | 2023 | arXiv |
| [Textbooks are all you need](https://doi.org/10.48550/arXiv.2306.11644) | 2023 | arXiv |
| [Textbooks are all you need ii: phi-1.5 technical report](https://arxiv.org/abs/2309.05463) | 2023 | arXiv |
| [Tinystories: How small can language models be and still speak coherent english?](https://doi.org/10.48550/arXiv.2305.07759) | 2023 | arXiv |
| [Enhancing multilingual language model with massive multilingual knowledge triples](https://arxiv.org/abs/2111.10962) | 2021 | arXiv |

#### Domain Model Enhancement

**Reasoning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute zero: Reinforced self-play reasoning with zero data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [HS-STAR: Hierarchical Sampling for Self-Taught Reasoners via Difficulty Estimation and Budget Reallocation](https://arxiv.org/abs/2505.19866) | 2025 | arXiv |
| [Learning to Pose Problems: Reasoning-Driven and Solver-Adaptive Data Synthesis for Large Reasoning Models](https://arxiv.org/abs/2511.09907) | 2025 | arXiv |
| [Logictree: Improving complex reasoning of LLMs via instantiated multi-step synthetic logical data](https://openreview.net/pdf?id=z4AMrCOetn) | 2025 | NeurIPS |
| [Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2505.20161) | 2025 | arXiv |
| [Seed-Coder: Let the Code Model Curate Data for Itself](https://arxiv.org/abs/2506.03524) | 2025 | arXiv |
| [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models](https://arxiv.org/abs/2505.00979) | 2025 | arXiv |
| [Thinking LLMs: General Instruction Following with Thought Generation](https://doi.org/10.48550/arXiv.2410.10630) | 2025 | ICML |
| [Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch](https://doi.org/10.48550/arXiv.2410.18693) | 2025 | arXiv |
| [A Graph-Based Synthetic Data Pipeline for Scaling High-Quality Reasoning Instructions](https://arxiv.org/abs/2412.08864) | 2024 | arXiv |
| [Aligning teacher with student preferences for tailored training data generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [Autocoder: Enhancing code large language model with$\backslash$textsc $$AIEV-Instruct$$](https://arxiv.org/abs/2405.14906) | 2024 | arXiv |
| [Boosting reward model with preference-conditional multi-aspect synthetic data generation](https://arxiv.org/abs/2407.16008) | 2024 | arXiv |
| [ControlMath: Controllable Data Generation Promotes Math Generalist Models](https://doi.org/10.18653/v1/2024.emnlp-main.680) | 2024 | EMNLP |
| [From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis](https://doi.org/10.18653/v1/2024.emnlp-main.284) | 2024 | EMNLP |
| [HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data](https://arxiv.org/abs/2409.06446) | 2024 | arXiv |
| [Infinitymath: A scalable instruction tuning dataset in programmatic mathematical reasoning](https://doi.org/10.1145/3627673.3679122) | 2024 | CIKM |
| [Jiuzhang3. 0: Efficiently improving mathematical reasoning by training small data synthesis models](http://papers.nips.cc/paper_files/paper/2024/hash/0356216f73660e15670510f5e42b5fa6-Abstract-Conference.html) | 2024 | NeurIPS |
| [MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning](https://openreview.net/forum?id=yLClGs770I) | 2024 | ICLR |
| [Marco-o1: Towards open reasoning models for open-ended solutions](https://arxiv.org/abs/2411.14405) | 2024 | arXiv |
| [MathScale: Scaling Instruction Tuning for Mathematical Reasoning](https://openreview.net/forum?id=Kjww7ZN47M) | 2024 | ICML |
| [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://openreview.net/forum?id=N8N0hgNDRt) | 2024 | ICLR |
| [Openmathinstruct-1: A 1.8 million math instruction tuning dataset](http://papers.nips.cc/paper_files/paper/2024/hash/3d5aa9a7ce28cdc710fbd044fd3610f3-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | NeurIPS |
| [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://doi.org/10.48550/arXiv.2403.09629) | 2024 | COLM |
| [Refined direct preference optimization with synthetic data for behavioral alignment of llms](https://arxiv.org/abs/2402.08005) | 2024 | arXiv |
| [Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold](http://papers.nips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html) | 2024 | NeurIPS |
| [Self-Consistency Preference Optimization](https://arxiv.org/abs/2411.04109) | 2024 | arXiv |
| [Self-play with execution feedback: Improving instruction-following capabilities of large language models](https://arxiv.org/abs/2406.13542) | 2024 | arXiv |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | arXiv |
| [Small Language Models Need Strong Verifiers to Self-Correct Reasoning](https://doi.org/10.18653/v1/2024.findings-acl.924) | 2024 | ACL |
| [Strengthening multimodal large language model with bootstrapped preference optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving](https://openreview.net/forum?id=Ep0TtjVoap) | 2024 | ICLR |
| [Tree-instruct: A preliminary study of the intrinsic relationship between complexity and alignment](https://aclanthology.org/2024.lrec-main.1460) | 2024 | COLING |
| [WizardLM: Empowering large pre-trained language models to follow complex instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |
| [Lamini-lm: A diverse herd of distilled models from large-scale instructions](https://arxiv.org/abs/2304.14402) | 2023 | arXiv |
| [Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning](https://arxiv.org/abs/2310.03731) | 2023 | arXiv |
| [Reflection-tuning: Recycling data for better instruction-tuning](https://openreview.net/forum?id=xaqoZZqkPU&utm_source=ainews&utm_medium=email&utm_campaign=ainews-reflection-70b-by-matt-from-it-department) | 2023 | NeurIPS |
| [What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning](https://arxiv.org/abs/2312.15685) | 2023 | arXiv |
| [Wizardcoder: Empowering code large language models with evol-instruct](https://arxiv.org/abs/2306.08568) | 2023 | arXiv |
| [Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct](https://arxiv.org/abs/2308.09583) | 2023 | arXiv |
| [Star: Bootstrapping reasoning with reasoning](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | 2022 | NeurIPS |

**Code**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Absolute zero: Reinforced self-play reasoning with zero data](https://arxiv.org/abs/2505.03335) | 2025 | arXiv |
| [Case2Code: Scalable Synthetic Data for Code Generation](https://aclanthology.org/2025.coling-main.733/) | 2025 | COLING |
| [Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks](https://arxiv.org/abs/2510.23208) | 2025 | arXiv |
| [Seed-Coder: Let the Code Model Curate Data for Itself](https://arxiv.org/abs/2506.03524) | 2025 | arXiv |
| [Autocoder: Enhancing code large language model with$\backslash$textsc $$AIEV-Instruct$$](https://arxiv.org/abs/2405.14906) | 2024 | arXiv |
| [Codeclm: Aligning language models with tailored synthetic data](https://arxiv.org/abs/2404.05875) | 2024 | arXiv |
| [HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data](https://arxiv.org/abs/2409.06446) | 2024 | arXiv |
| [WizardLM: Empowering large pre-trained language models to follow complex instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |
| [Wizardcoder: Empowering code large language models with evol-instruct](https://arxiv.org/abs/2306.08568) | 2023 | arXiv |

**Instruction following**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Recursive introspection: Teaching language model agents how to self-improve](http://papers.nips.cc/paper_files/paper/2024/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html) | 2024 | NeurIPS |
| [Selective reflection-tuning: Student-selected data recycling for llm instruction-tuning](https://doi.org/10.18653/v1/2024.findings-acl.958) | 2024 | ACL |
| [Self-play with execution feedback: Improving instruction-following capabilities of large language models](https://arxiv.org/abs/2406.13542) | 2024 | arXiv |
| [Self-refine instruction-tuning for aligning reasoning in language models](https://arxiv.org/abs/2405.00402) | 2024 | arXiv |
| [Spar: Self-play with tree-search refinement to improve instruction-following in large language models](https://arxiv.org/abs/2412.11605) | 2024 | arXiv |
| [Tree-instruct: A preliminary study of the intrinsic relationship between complexity and alignment](https://aclanthology.org/2024.lrec-main.1460) | 2024 | LREC-COLING |
| [WizardLM: Empowering large pre-trained language models to follow complex instructions](https://openreview.net/forum?id=CfXh93NDgH) | 2024 | ICLR |
| [\# instag: Instruction tagging for analyzing supervised fine-tuning of large language models](https://arxiv.org/abs/2308.07074) | 2023 | arXiv |
| [Alpagasus: Training a better alpaca with fewer data](https://arxiv.org/abs/2307.08701) | 2023 | arXiv |
| [Lamini-lm: A diverse herd of distilled models from large-scale instructions](https://arxiv.org/abs/2304.14402) | 2023 | arXiv |
| [Lima: Less is more for alignment](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) | 2023 | NeurIPS |
| [Reflection-tuning: Recycling data for better instruction-tuning](https://openreview.net/forum?id=xaqoZZqkPU&utm_source=ainews&utm_medium=email&utm_campaign=ainews-reflection-70b-by-matt-from-it-department) | 2023 | NeurIPS |
| [What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning](https://arxiv.org/abs/2312.15685) | 2023 | arXiv |
| [Self-instruct: Aligning language models with self-generated instructions](https://arxiv.org/abs/2212.10560) | 2022 | arXiv |

**Preference**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](https://openreview.net/forum?id=BPgK5XW1Nb) | 2025 | ICLR |
| [Aligning teacher with student preferences for tailored training data generation](https://arxiv.org/abs/2406.19227) | 2024 | arXiv |
| [Boosting reward model with preference-conditional multi-aspect synthetic data generation](https://arxiv.org/abs/2407.16008) | 2024 | arXiv |
| [Course-correction: Safety alignment using synthetic preferences](https://arxiv.org/abs/2407.16637) | 2024 | arXiv |
| [Refined direct preference optimization with synthetic data for behavioral alignment of llms](https://doi.org/10.1007/978-3-031-82481-4_7) | 2024 | arXiv |
| [Self-Consistency Preference Optimization](https://arxiv.org/abs/2411.04109) | 2024 | arXiv |
| [Self-directed synthetic dialogues and revisions technical report](https://arxiv.org/abs/2407.18421) | 2024 | arXiv |
| [Strengthening multimodal large language model with bootstrapped preference optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |

**In-context learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Embracing Collaboration Over Competition: Condensing Multiple Prompts for Visual In-Context Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Embracing_Collaboration_Over_Competition_Condensing_Multiple_Prompts_for_Visual_In-Context_CVPR_2025_paper.html) | 2025 | CVPR |
| [Self-Harmonized Chain of Thought](https://doi.org/10.18653/v1/2025.naacl-long.53) | 2025 | arXiv |
| [Auto-ICL: In-Context Learning without Human Supervision](https://doi.org/10.48550/arXiv.2311.09263) | 2024 | arXiv |
| [Demonstration Augmentation for Zero-shot In-context Learning](https://doi.org/10.18653/v1/2024.findings-acl.846) | 2024 | ACL |
| [Self-Prompting Large Language Models for Zero-Shot Open-Domain QA](https://doi.org/10.18653/v1/2024.naacl-long.17) | 2024 | arXiv |
| [Automatic Chain of Thought Prompting in Large Language Models](https://openreview.net/forum?id=5NTt8GFjUHkr) | 2023 | ICLR |
| [Generate Rather than Retrieve: Large Language Models Are Strong Context Generators](https://openreview.net/forum?id=fB0hRu9GZUS) | 2023 | arXiv |
| [Privacy-preserving in-context learning with differentially private few-shot generation](https://arxiv.org/abs/2309.11765) | 2023 | arXiv |
| [Recitation-Augmented Language Models](https://openreview.net/forum?id=-cqvvvb-NkI) | 2023 | arXiv |
| [Self-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations](https://doi.org/10.18653/v1/2023.emnlp-main.968) | 2023 | EMNLP |
| [Self-Generated In-Context Learning: Leveraging Auto-regressive Language Models as a Demonstration Generator](https://doi.org/10.48550/arXiv.2206.08082) | 2022 | arXiv |

**Reinforcement Learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Kimi K2: Open Agentic Intelligence](https://doi.org/10.48550/arXiv.2507.20534) | 2025 | arXiv |
| [s1: Simple test-time scaling](https://doi.org/10.48550/arXiv.2501.19393) | 2025 | arXiv |
| [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://doi.org/10.48550/arXiv.2504.04736) | 2025 | arXiv |
| [Synthetic Data RL: Task Definition Is All You Need](https://doi.org/10.48550/arXiv.2505.17063) | 2025 | arXiv |
| [RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold](http://papers.nips.cc/paper_files/paper/2024/hash/4b77d5b896c321a29277524a98a50215-Abstract-Conference.html) | 2024 | arXiv |

#### Model Evaluation

**Synthetic benchmark**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Achilles' Heel of Mamba: Essential difficulties of the Mamba architecture demonstrated by synthetic data](https://arxiv.org/abs/2509.17514) | 2025 | arXiv |
| [Measuring General Intelligence with Generated Games](https://doi.org/10.48550/arXiv.2505.07215) | 2025 | arXiv |
| [Efficacy of Synthetic Data as a Benchmark](https://arxiv.org/abs/2409.11968) | 2024 | arXiv |
| [Can you rely on your model evaluation? improving model evaluation with synthetic test data](http://papers.nips.cc/paper_files/paper/2023/hash/05fb0f4e645cad23e0ab59d6b9901428-Abstract-Conference.html) | 2023 | NeurIPS |
| [SynBench: Task-Agnostic Benchmarking of Pretrained Representations using Synthetic Data](https://doi.org/10.48550/arXiv.2210.02989) | 2022 | NeurIPS |
---

### 2.3 Trustworthy AI

#### Privacy

**Privacy-preserving learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Does Training with Synthetic Data Truly Protect Privacy?](https://arxiv.org/abs/2502.12976) | 2025 | arXiv |
| [Federated Knowledge Recycling: Privacy-preserving synthetic data sharing](https://arxiv.org/abs/2407.20830) | 2025 | PATTERN RECOGN LETT |
| [Synthetic image learning: Preserving performance and preventing membership inference attacks](https://doi.org/10.48550/arXiv.2407.15526) | 2025 | PATTERN RECOGN LETT |
| [Differentially Private Synthetic Data via Foundation Model APIs 1: Images](https://arxiv.org/abs/2305.15560)) | 2024 | ICLR |
| [Differentially private synthetic data via foundation model APIs 2: Text](https://arxiv.org/abs/2403.01749) | 2024 | arXiv |
| [Enhancing the utility of privacy-preserving cancer classification using synthetic data](https://doi.org/10.1007/978-3-031-77789-9_6) | 2024 | MICCAI |
| [Generate synthetic text approximating the private distribution with differential privacy](https://www.ijcai.org/proceedings/2024/735) | 2024 | IJCAI |
| [Privacy-preserving instructions for aligning large language models](https://arxiv.org/abs/2402.13659) | 2024 | arXiv |
| [SETSUBUN: Revisiting Membership Inference Game for Evaluating Synthetic Data Generation](https://doi.org/10.2197/ipsjjip.32.757) | 2024 | J INF PROCESS SYST |
| [Synthetic data aided federated learning using foundation models](https://doi.org/10.48550/arXiv.2407.05174) | 2024 | arXiv |
| [Towards Privacy-Preserving Relational Data Synthesis via Probabilistic Relational Models](https://doi.org/10.1007/978-3-031-70893-0_13) | 2024 | German Conference on Artificial Intellig |
| [Using synthetic data to mitigate unfairness and preserve privacy through single-shot federated learning](https://doi.org/10.48550/arXiv.2409.09532) | 2024 | arXiv |
| [VertiMRF: Differentially Private Vertical Federated Data Synthesis](https://doi.org/10.1145/3637528.3671771) | 2024 | KDD |

**Model inversion attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Gradient inversion of federated diffusion models](https://arxiv.org/abs/2405.20380) | 2024 | arXiv |
| [Understanding Data Reconstruction Leakage in Federated Learning from a Theoretical Perspective](https://arxiv.org/abs/2408.12119) | 2024 | arXiv |
| [The secret revealer: Generative model-inversion attacks against deep neural networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_The_Secret_Revealer_Generative_Model-Inversion_Attacks_Against_Deep_Neural_Networks_CVPR_2020_paper.html) | 2020 | CVPR |
| [A methodology for formalizing model-inversion attacks](https://doi.org/10.1109/CSF.2016.32) | 2016 | CSF |
| [Model inversion attacks that exploit confidence information and basic countermeasures](https://doi.org/10.1145/2810103.2813677) | 2015 | SIGSAC |

#### Safety & Security

**Model stealing attack**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging](https://arxiv.org/abs/2505.19892) | 2025 | arXiv |
| [Data-free model extraction](https://doi.org/10.48550/arXiv.2507.16969) | 2021 | CVPR |
| [Maze: Data-free model stealing attack using zeroth-order gradient estimation](https://openaccess.thecvf.com/content/CVPR2021/html/Kariyappa_MAZE_Data-Free_Model_Stealing_Attack_Using_Zeroth-Order_Gradient_Estimation_CVPR_2021_paper.html) | 2021 | CVPR |

**Adversarial defense**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Deceptive diffusion: generating synthetic adversarial examples](https://doi.org/10.1007/978-3-031-92366-1_25) | 2025 | SSVM |
| [Is synthetic data all we need? benchmarking the robustness of models trained with synthetic images](https://doi.org/10.1109/CVPRW63382.2024.00257) | 2024 | CVPR |
| [TSynD: Targeted Synthetic Data Generation for Enhanced Medical Image Classification: Leveraging Epistemic Uncertainty to Improve Model Performance](https://doi.org/10.1007/978-3-658-47422-5_35) | 2024 | arXiv |
| [Leaving reality to imagination: Robust classification via generated datasets](https://arxiv.org/abs/2302.02503) | 2023 | arXiv |
| [Robust learning meets generative models: Can proxy distributions improve adversarial robustness?](https://arxiv.org/abs/2104.09425) | 2021 | arXiv |
| [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572) | 2014 | arXiv |

**Machine unlearning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal](https://arxiv.org/abs/2403.01244) | 2024 | arXiv |

#### Fairness

**De-bias learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Generating Informative Samples for Risk-Averse Fine-Tuning of Downstream Tasks](https://openreview.net/forum?id=kfB5Ciz2XZ) | 2025 | NeurIPS |
| [Know" No''Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP](https://arxiv.org/abs/2501.10913) | 2025 | arXiv |
| [Modeling multi-task model merging as adaptive projective gradient descent](https://doi.org/10.48550/arXiv.2501.01230) | 2025 | ICML |
| [Virus Infection Attack on LLMs: Your Poisoning Can Spread" VIA" Synthetic Data](https://arxiv.org/abs/2509.23041) | 2025 | arXiv |
| [FADE: Towards Fairness-aware Generation for Domain Generalization via Classifier-Guided Score-based Diffusion Models](https://arxiv.org/abs/2406.09495) | 2024 | arXiv |
| [Refined direct preference optimization with synthetic data for behavioral alignment of llms](https://doi.org/10.1007/978-3-031-82481-4_7) | 2024 | ICML |
| [Strengthening multimodal large language model with bootstrapped preference optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [SynthFair: Ensuring Subgroup Fairness in Classification via Synthetic Data Generation](https://link.springer.com/chapter/10.1007/978-3-031-85856-7_26) | 2024 | CSCE |
| [Using synthetic data to mitigate unfairness and preserve privacy through single-shot federated learning](https://doi.org/10.48550/arXiv.2409.09532) | 2024 | arXiv |
| [Discover and cure: Concept-aware mitigation of spurious correlation](https://proceedings.mlr.press/v202/wu23w.html) | 2023 | ICML |

**Long-tail learning**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature](https://arxiv.org/abs/2405.04819) | 2024 | arXiv |
| [Ltgc: Long-tail recognition via leveraging llms-driven generated content](https://doi.org/10.1109/CVPR52733.2024.01845) | 2024 | CVPR |
| [Feedback-guided data synthesis for imbalanced classification](https://arxiv.org/abs/2310.00158) | 2023 | arXiv |

#### Interpretability

**Explainable AI**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Debiasing synthetic data generated by deep generative models](http://papers.nips.cc/paper_files/paper/2024/hash/4902603fe8cb095b9ada707a19bd151c-Abstract-Conference.html) | 2024 | NeurIPS |
| [Knowledge-driven AI-generated data for accurate and interpretable breast ultrasound diagnoses](https://arxiv.org/abs/2407.16634) | 2024 | arXiv |
| [Don't trust your eyes: on the (un) reliability of feature visualizations](https://arxiv.org/abs/2306.04719) | 2023 | arXiv |
| [Multifaceted feature visualization: Uncovering the different types of features learned by each neuron in deep neural networks](https://arxiv.org/abs/1602.03616) | 2016 | arXiv |
| [Understanding deep image representations by inverting them](https://doi.org/10.1109/CVPR.2015.7299155) | 2015 | CVPR |

#### Governance

**Data watermarking**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Can watermarking large language models prevent copyrighted text generation and hide training data?](https://doi.org/10.1609/aaai.v39i23.34684) | 2025 | AAAI |
| [TimeWak: Temporal Chained-Hashing Watermark for Time Series Data](https://arxiv.org/abs/2506.06407) | 2025 | arXiv |
---

### 2.4 Embodied AI

#### Perception

**Visual sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Re$^3$Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation](https://arxiv.org/abs/2502.08645) | 2025 | arXiv |
| [Splatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting](https://doi.org/10.1109/ICRA55743.2025.11128339) | 2025 | ICRA |
| [Habitat 3.0: A Co-Habitat for Humans, Avatars, and Robots](https://openreview.net/forum?id=4znwzG92CE) | 2024 | ICLR |
| [Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation](https://doi.org/10.1109/CVPR52733.2024.01550) | 2024 | CVPR |
| [Dextreme: Transfer of agile in-hand manipulation from simulation to reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |
| [Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments](https://doi.org/10.1109/LRA.2023.3270034) | 2023 | RAL |
| [ProcTHOR: Large-Scale Embodied AI Using Procedural Generation](http://papers.nips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html) | 2022 | NeurIPS |
| [MuJoCo: A physics engine for model-based control](https://doi.org/10.1109/IROS.2012.6386109) | 2012 | IROS |

**Force sensing**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695) | 2024 | arXiv |
| [ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes](https://doi.org/10.1109/ICCV51070.2023.01873) | 2023 | ICCV |
| [Dextreme: Transfer of agile in-hand manipulation from simulation to reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |

**Sensor fusion**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://doi.org/10.1109/CVPR52733.2024.01370) | 2024 | CVPR |
| [Embodiedgpt: Vision-language pre-training via embodied chain of thought](http://papers.nips.cc/paper_files/paper/2023/hash/4ec43957eda1126ad4887995d05fae3b-Abstract-Conference.html) | 2023 | NeurIPS |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | 2023 | arXiv |
| [Rt-2: Vision-language-action models transfer web knowledge to robotic control](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | CoRL |

#### Interaction

**Trajectory synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning](https://doi.org/10.1109/ICRA55743.2025.11127809) | 2025 | ICRA |
| [ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis](https://arxiv.org/abs/2503.14526) | 2025 | arXiv |
| [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](https://doi.org/10.15607/RSS.2024.XX.067) | 2024 | RSS |
| [Intervengen: Interventional data generation for robust and data-efficient robot imitation learning](https://doi.org/10.1109/IROS58592.2024.10801523) | 2024 | IROS |
| [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://doi.org/10.1177/02783649241273668) | 2023 | RSS |
| [MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations](https://proceedings.mlr.press/v229/mandlekar23a.html) | 2023 | CoRL |
| [Scaling robot learning with semantically imagined experience](https://arxiv.org/abs/2302.11550) | 2023 | arXiv |

**Environment synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Holodeck: Language guided generation of 3d embodied ai environments](https://doi.org/10.1109/CVPR52733.2024.01536) | 2024 | CVPR |
| [Partnr: A benchmark for planning and reasoning in embodied multi-agent tasks](https://arxiv.org/abs/2411.00081) | 2024 | arXiv |
| [RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation](https://openreview.net/forum?id=SQIDlJd3hN) | 2024 | ICML |
| [Teach: Task-driven embodied agents that chat](https://doi.org/10.1609/aaai.v36i2.20097) | 2022 | AAAI |
| [Luminous: Indoor scene generation for embodied ai challenges](https://arxiv.org/abs/2111.05527) | 2021 | arXiv |
| [Alfred: A benchmark for interpreting grounded instructions for everyday tasks](https://openaccess.thecvf.com/content_CVPR_2020/html/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.html) | 2020 | CVPR |
| [Virtualhome: Simulating household activities via programs](http://openaccess.thecvf.com/content_cvpr_2018/html/Puig_VirtualHome_Simulating_Household_CVPR_2018_paper.html) | 2018 | CVPR |

**Human behavior synthesis**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [3D Human Reconstruction in the Wild with Synthetic Data Using Generative Models](https://arxiv.org/abs/2403.11111) | 2024 | arXiv |
| [Social-llava: Enhancing robot navigation through human-language reasoning in social spaces](https://arxiv.org/abs/2501.09024) | 2024 | arXiv |
| [Toward human-like social robot navigation: A large-scale, multi-modal, social human navigation dataset](https://doi.org/10.1109/IROS55552.2023.10342447) | 2023 | IROS |
| [Socially compliant navigation dataset (scand): A large-scale dataset of demonstrations for social navigation](https://doi.org/10.1109/LRA.2022.3184025) | 2022 | RAL |

#### Generalization

**Cross-embodiment training**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Droid: A large-scale in-the-wild robot manipulation dataset](https://arxiv.org/abs/2403.12945) | 2024 | arXiv |
| [Octo: An open-source generalist robot policy](https://arxiv.org/abs/2405.12213) | 2024 | arXiv |
| [Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0](https://doi.org/10.48550/arXiv.2310.08864) | 2024 | ICRA |
| [Openvla: An open-source vision-language-action model](https://arxiv.org/abs/2406.09246) | 2024 | arXiv |

**Vision-language-action models**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Diffusion forcing: Next-token prediction meets full-sequence diffusion](http://papers.nips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html) | 2024 | NeurIPS |
| [Learning universal policies via text-guided video generation](http://papers.nips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html) | 2023 | NeurIPS |
| [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) | 2023 | arXiv |
| [Rt-2: Vision-language-action models transfer web knowledge to robotic control](https://proceedings.mlr.press/v229/zitkovich23a.html) | 2023 | CoRL |

**Sim-to-real transfer**
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Re$^3$Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation](https://arxiv.org/abs/2502.08645) | 2025 | arXiv |
| [Splatsim: Zero-shot sim2real transfer of rgb manipulation policies using gaussian splatting](https://doi.org/10.1109/ICRA55743.2025.11128339) | 2025 | ICRA |
| [Transic: Sim-to-real policy transfer by learning from online correction](https://arxiv.org/abs/2405.10315) | 2024 | arXiv |
| [Dextreme: Transfer of agile in-hand manipulation from simulation to reality](https://doi.org/10.1109/ICRA48891.2023.10160216) | 2023 | ICRA |
| [Bi-directional domain adaptation for sim2real transfer of embodied navigation agents](https://doi.org/10.1109/LRA.2021.3062303) | 2021 | RAL |
| [RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer](https://doi.org/10.1109/ICRA48506.2021.9561157) | 2020 | arXiv |

---

## 3. Challenges & Future Directions

### Model Collapse ###
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A theoretical perspective: How to prevent model collapse in self-consuming training loops](https://arxiv.org/abs/2502.18865) | 2025 | arXiv |
| [AI models collapse when trained on recursively generated data](https://doi.org/10.48550/arXiv.2410.12954) | 2024 | Nature |
| [Beyond model collapse: Scaling up with synthesized data requires verification](https://arxiv.org/abs/2406.07515) | 2024 | arXiv |
| [Fairness feedback loops: training on synthetic data amplifies bias](https://doi.org/10.1145/3630106.3659029) | 2024 | FACCT |
| [Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data](https://arxiv.org/abs/2404.01413) | 2024 | arXiv |
| [Self-correcting self-consuming loops for generative model training](https://arxiv.org/abs/2402.07087) | 2024 | arXiv |
| [Towards theoretical understandings of self-consuming generative models](https://arxiv.org/abs/2402.11778) | 2024 | arXiv |
| [Large language models suffer from their own output: An analysis of the self-consuming training loop](https://doi.org/10.48550/arXiv.2311.16822) | 2023 | arXiv |
| [On the stability of iterative retraining of generative models on their own data](https://arxiv.org/abs/2310.00429) | 2023 | arXiv |
| [Self-consuming generative models go mad](https://openreview.net/forum?id=ShjMHfmPs0) | 2023 | ICLR |

### Utility-Privacy Tradeoffs ###

### Generation-Evaluation Bias ###

### Active Data Synthesis ###
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Controlled training data generation with diffusion models](https://arxiv.org/abs/2403.15309) | 2024 | arXiv |
| [Llm see, llm do: Guiding data generation to target non-differentiable objectives](https://arxiv.org/abs/2407.01490) | 2024 | arXiv |

### Synthetic Data Evaluation ###
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [A multi-faceted evaluation framework for assessing synthetic data generated by large language models](https://arxiv.org/abs/2404.14445) | 2024 | arXiv |

### Multi-Modal Data Synthesis ###
| Paper Title | Year | Conference/Journal |
|-------------|------|--------------------|
| [Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data](https://doi.org/10.18653/v1/2024.findings-acl.864) | 2024 | ACL |
| [Strengthening multimodal large language model with bootstrapped preference optimization](https://doi.org/10.1007/978-3-031-73414-4_22) | 2024 | ECCV |
| [Synthvlm: High-efficiency and high-quality synthetic data for vision language models](https://doi.org/10.48550/arXiv.2407.20756) | 2024 | CoRL |
---

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
