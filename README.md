# XAI
## Papers and code of Explainable AI esp. w.r.t. Image classificiation


### 2013 Conference Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Visualization of CNN** | Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps | [CVPR2013](https://arxiv.org/pdf/1312.6034v2.pdf) | [PyTorch](https://github.com/idiap/fullgrad-saliency) | `Visualization gradient-based saliency maps`  |



### 2016 Conference Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **CAM** | [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/) | [CVPR2016](https://arxiv.org/pdf/1512.04150v1.pdf) | [PyTorch (Official)](https://github.com/zhoubolei/CAM) | `class activation mapping`  |
| **LIME** | [“Why Should I Trust You?”Explaining the Predictions of Any Classifier](https://homes.cs.washington.edu/~marcotcr/blog/lime/) | [KDD2016](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | [PyTorch (Official)](https://github.com/marcotcr/lime-experiments) | `trust a prediction`  |



### 2017 Conference Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Grad-CAM** | Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | [ICCV2017, CVPR2016 (original)](https://arxiv.org/pdf/1610.02391.pdf) | [PyTorch](https://github.com/jacobgil/pytorch-grad-cam) | `Visualization gradient-based saliency maps`  |
| **Network Dissection** | [Network Dissection: Quantifying Interpretability of Deep Visual Representations](http://netdissect.csail.mit.edu/) | [CVPR2017](https://arxiv.org/pdf/1704.05796.pdf) | [PyTorch (Official)](https://github.com/CSAILVision/NetDissect) | `Visualization`  |


### 2018 Conference Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **TCAV** | [Interpretability Beyond Feature Attribution:Quantitative Testing with Concept Activation Vectors (TCAV)](https://beenkim.github.io/) | [ICML 2018](https://arxiv.org/pdf/1711.11279.pdf) | [Tensorflow 1.15.2](https://github.com/tensorflow/tcav) | `interpretability method`  |
| **Interpretable CNN** | [Interpretable Convolutional Neural Networks](http://qszhang.com/index.php/icnn/) | [CVPR 2018](https://arxiv.org/pdf/1710.00935.pdf) | [Tensorflow 1.x](https://github.com/andrehuang/InterpretableCNN) | `explainability by design`  |
| **Anchors** | Anchors: High-Precision Model-Agnostic Explanations | [AAAI 2018](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf) | [sklearn (Official)](https://github.com/marcotcr/anchor-experiments) | `model-agnostic`  |
| **Sanity Checks** | [Sanity checks for saliency maps](http://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps) | [NeurIPS 2018](https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf) | [PyTorch](https://github.com/jendawkins/saliencySanity) | `saliency methods vs edge detector`  |
| **Grad Cam++** | [Grad Cam++:Improved Visual Explanations forDeep Convolutional Networks](https://ieeexplore.ieee.org/document/8354201) | [WACV 2018](https://arxiv.org/pdf/1710.11063.pdf) | [PyTorch](https://github.com/1Konny/gradcam_plus_plus-pytorch) | `saliency maps`  |
|**Interpretable Basis**|Interpretable Basis Decomposition for Visual Explanation|[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Antonio_Torralba_Interpretable_Basis_Decomposition_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/CSAILVision/IBD)|`ibd`|

### 2019 Conference Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Full-grad** | [Full-Gradient Representation for Neural Network Visualization](https://www.idiap.ch/workshop/smld2019/slides/smld2019_suraj_srinivas.pdf) | [NeurIPS2019](https://arxiv.org/pdf/1905.00780.pdf) | [PyTorch (Official)](https://github.com/idiap/fullgrad-saliency) [Tensorflow](https://github.com/vk1996/fullgradsaliency_tf) | `saliency map representation`  |
| **This looks like that** | [This Looks Like That: Deep Learning for Interpretable Image Recognition](https://papers.nips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition) | [NeurIPS2019](https://papers.nips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf) | [PyTorch (Official)](https://github.com/cfchen-duke/ProtoPNet) | `object`  |
| **Counterfactual visual explanations** | Counterfactual visual explanations | [ICML2019](https://arxiv.org/pdf/1904.07451v2.pdf) |  | `interpretability`  |
|**concept with contribution interpretable cnn**|Explaining Neural Networks Semantically and Quantitatively|[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Explaining_Neural_Networks_Semantically_and_Quantitatively_ICCV_2019_paper.pdf)|||
|**SIS**|[What made you do this? Understanding black-box decisions with sufficient input subsets](http://proceedings.mlr.press/v89/carter19a/carter19a.pdf)|[AISTATS 2019 - Supplementary Material](http://proceedings.mlr.press/v89/carter19a/carter19a-supp.pdf)|[Tensorflow 1.x](https://github.com/b-carter/SufficientInputSubsets)||
|**Filter as concept detector**|Filters in Convolutional Neural Networks as Independent Detectors of Visual Concepts|[ACM](https://dl.acm.org/doi/10.1145/3345252.3345294)|||

### 2020 Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **INN** | [Making Sense of CNNs: Interpreting Deep Representations & Their Invariances with INNs](https://compvis.github.io/invariances/)  | [ECCV 2020](https://arxiv.org/pdf/2008.01777.pdf) | [PyTorch](https://github.com/CompVis/invariances) | `explainability by design` |
| **X-Grad CAM** | [Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs ](https://arxiv.org/pdf/2008.02312.pdf) | | [PyTorch](https://github.com/Fu0511/XGrad-CAM) | |
| **Revisiting BP saliency** | [There and Back Again: Revisiting Backpropagation Saliency Methods](https://arxiv.org/pdf/2004.02866.pdf)| [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Rebuffi_There_and_Back_Again_Revisiting_Backpropagation_Saliency_Methods_CVPR_2020_paper.pdf)| [PyTorch](https://github.com/srebuffi/revisiting_saliency)|`grad cam failure noted`|
|**Interacting with explanation**| [Making deep neural networks right for the right scientific reasons by interacting with their explanations](https://arxiv.org/pdf/2001.05371.pdf)| [Nature Machine Intelligence](https://www.nature.com/articles/s42256-020-0212-3)|[sklearn](https://codeocean.com/capsule/7818629/tree/v1)||
|**Class specific Filters**|[Training Interpretable Convolutional Neural Networks by Differentiating Class-specific Filters](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470613.pdf)|[ECCV Supplementary Material](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470613-supp.pdf)|[Code - not yet updated](https://github.com/hyliang96/CSGCNN)|[ICLR rejected version with reviews](https://openreview.net/forum?id=r1ltnp4KwS)|
|**Interpretable Decoupling**|Interpretable Neural Network Decoupling|[ECCV 2020](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600647.pdf)|||
|**iCaps**|[iCaps: An Interpretable Classifier via Disentangled Capsule Networks](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640307.pdf)|[ECCV Supplementary Material](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640307-supp.pdf)|||
|**VQA**|Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision|[ECCV 2020](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540528.pdf)|[PyTorch](https://github.com/GeraldHan/TRN)||
|**When explanations lie**|When Explanations Lie: Why Many Modified BP Attributions Fail|[ICML 2020](https://arxiv.org/pdf/1912.09818v6.pdf)|[PyTorch](https://github.com/berleon/when-explanations-lie)||
|**Similarity models**|Towards Visually Explaining Similarity Models| [Arxiv](https://arxiv.org/pdf/2008.06035.pdf)|||
|**Quantify trust**|How Much Should I Trust You? Modeling Uncertainty of Black Box Explanations|[NeurIPS 2020 submission](https://arxiv.org/pdf/2008.05030.pdf)||`hima_lakkaraju`,`sameer_singh`,`model-agnostic`|
|**Concepts for segmentation task**|ABSTRACTING DEEP NEURAL NETWORKS INTO CONCEPT GRAPHS FOR CONCEPT LEVEL INTERPRETABILITY| [Arxiv](https://arxiv.org/pdf/2008.06457.pdf)|[Tensorflow 1.14](https://github.com/koriavinash1/BioExp)|`brain tumour segmentation`|
|**Deep Lift based Network Pruning**|Utilizing Explainable AI for Quantization and Pruning of Deep Neural Networks|[Arxiv NeurIPS format](https://arxiv.org/pdf/2008.09072.pdf)||`nas`,`deep_lift`|
|**Unifed Attribution Framework**|A Unified Taylor Framework for Revisiting Attribution Methods|[Arxiv](https://arxiv.org/pdf/2008.09695.pdf)||`taylor`,`attribution_framework`|
|**Global Cocept Attribution**|Towards Global Explanations of Convolutional Neural Networks with Concept Attribution|[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Towards_Global_Explanations_of_Convolutional_Neural_Networks_With_Concept_Attribution_CVPR_2020_paper.pdf)|||
|**relevance estimation**|Determining the Relevance of Features for Deep Neural Networks|[ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710324.pdf)|||
|**localized concept maps**|Explaining AI-based Decision Support Systems using Concept Localization Maps|[Arxiv](https://arxiv.org/pdf/2005.01399.pdf)|[Just repository created](https://github.com/adriano-lucieri/SCDB)||
|**quantify saliency**|Quantifying Explainability of Saliency Methods in Deep Neural Networks|[Arxiv](https://arxiv.org/pdf/2009.02899.pdf)|[PyTorch](https://github.com/etjoa003/explainable_ai/tree/master/xai_basic)||
|**generalization of LIME - MeLIME**|MeLIME: Meaningful Local Explanation for Machine Learning Models|[Arxiv](https://arxiv.org/pdf/2009.05818.pdf)|[Tensorflow 1.15](https://github.com/tiagobotari/melime)||
|**global counterfactual explanations**|Interpretable and Interactive Summaries of Actionable Recourses|[Arxiv](https://arxiv.org/pdf/2009.07165.pdf)|||
|**fine grained counterfactual heatmaps**|SCOUTER: Slot Attention-based Classifier for Explainable Image Recognition|[Arxiv](https://arxiv.org/pdf/2009.06138.pdf)|[PyTorch](https://github.com/wbw520/scouter)|`scouter`|
|**quantify trust**|How Much Can We Really Trust You? Towards Simple, Interpretable Trust Quantification Metrics for Deep Neural Networks|[Arxiv](https://arxiv.org/pdf/2009.05835.pdf)|||
|**Non-negative concept activation vectors**|IMPROVING INTERPRETABILITY OF CNN MODELS USING NON-NEGATIVE CONCEPT ACTIVATION VECTORS|[Arxiv](https://arxiv.org/pdf/2006.15417.pdf)|||
|**different layer activations**|Explaining Neural Networks by Decoding Layer Activations|[Arxiv](https://arxiv.org/pdf/2005.13630.pdf)|||
|**concept bottleneck networks**|Concept Bottleneck Models|[ICML 2020](https://arxiv.org/pdf/2007.04612.pdf)|[PyTorch](https://github.com/yewsiang/ConceptBottleneck)||
|**attribution**|Visualizing the Impact of Feature Attribution Baselines|[Distill](https://distill.pub/2020/attribution-baselines/)|||
|**CSI**|Contextual Semantic Interpretability|[Arxiv](https://arxiv.org/pdf/2009.08720.pdf)||`explainable_by_design`|
|**Improve black box via explanation**|Introspective Learning by Distilling Knowledge from Online Self-explanation|[Arxiv](https://arxiv.org/pdf/2009.09140.pdf)||`kowledge_distillation`|
|**Patch explanations**|Information-Theoretic Visual Explanation for Black-Box Classifiers|[Arxiv](https://arxiv.org/pdf/2009.11150.pdf)|[Tensorflow 1.13.1](https://github.com/nuclearboy95/XAI-Information-Theoretic-Explanation)|`patch_sampling`,`information_theory`|
### 2021 Papers
| Title | Paper Title                                                                                                 | Source Link                                                                                                                                                                                            | Code                                                                                          | Tags      
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
|**Debiasing concepts** | Debiasing Concept Bottleneck Models with Instrumental Variables|[ICLR 2021 submissions page](https://openreview.net/forum?id=6puUoArESGp)| |`causality`|
|**Prototype Trajectory**|Interpretable Sequence Classification Via Prototype Trajectory|[ICLR 2021 submissions page](https://openreview.net/forum?id=KwgQn_Aws3_)| |`this looks like that styled RNN`|
|**Shapley dependence assumption**|Shapley explainability on the data manifold|[ICLR 2021 submissions page](https://openreview.net/forum?id=OPyWRrcjVQw)|||
|**High dimension Shapley**|Human-interpretable model explainability on high-dimensional data|[ICLR 2021 submissions page](https://openreview.net/forum?id=VlRqY4sV9FO)|||
|**L2x like paper**|A Learning Theoretic Perspective on Local Explainability|[ICLR 2021 submissions page](https://openreview.net/forum?id=7aL-OtQrBWD)|||
|**Evaluation**|Evaluation of Similarity-based Explanations|[ICLR 2021 submissions page](https://openreview.net/forum?id=9uvhpyQwzM_)||`like adebayo paper for this looks like that styled methods`|
|**Model correction**|Defuse: Debugging Classifiers Through Distilling Unrestricted Adversarial Examples|[ICLR 2021 submissions page](https://openreview.net/pdf?id=3R--2TdxMps)|||
|**Subspace explanation**|Constraint-Driven Explanations of Black-Box ML Models|[ICLR 2021 submissions page](https://openreview.net/forum?id=kVZ6WBYazFq)||`to see how close to MUSE by Hima Lakkaraju 2019`|
