## Papers related to generated code detection

Regarding the rising era of LLMs, it is become challenging to develop tools and approaches to detect generated contents. The application is various specially in the education field to attest the autorship of the student's works, in the scientific domain, etc. In this repository, we gather all papers related to generated text detection and specially on code detection.

The following repository will be structured as sub-folders following these topics :

- [🌟 Generated Code Detection](#intro)
- [📜 Papers](#papers)
    - [🏷️ Tagset](#tagset)
    - [🎯 The List](#list)
- [🧰 Resources](#resources)
    - [📊 Datasets](#datasets)
    - [🛠️ Tools](#tools)


<a id="intro"></a>
## 🌟 Generated Code Detection


<a id="papers"></a>
## 📜 Papers

<a id="tagset"></a>
### 🏷️ Tagset

In this paper list, we tag each paper with one or more labels defined in the table below. These tags serve the purpose of facilitating the related work searching.

| Category | Explanation |
|----------|-------------|
| ![](https://img.shields.io/badge/Analysis-green) | The paper propose a analyse of the existing approach on the generated code detection and the potential features to consider to distinguish human writing from generated code.*. |
| ![](https://img.shields.io/badge/Machine_Learning-orange) | The paper proposed an approach based on Machine Learning applied to code features classification following the nature of the submited code.* |
| ![](https://img.shields.io/badge/Watermarking-cyan) | The paper proposed an approach based on code watermarking. These approches will imply to modify the generated of to force the model to adopt a certain code stylometric that can be detected.* |
| ![](https://img.shields.io/badge/Code_Protection-purple) | The paper proposed an approach to protect source code copyright by detecting the unauthorized used of a code dataset to train a model.* |


<a id="list"></a>
### 🎯 The List

> [!Note]
> The list is sorted by the date of the first time the paper was released.

1. **Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers** (ICSE 2025) ![](https://img.shields.io/badge/Analysis-green) <br />
    *Yuling Shi, Hongyu Zhang, Chengcheng Wan, Xiaodong Gu*
    [[paper](https://www.computer.org/csdl/proceedings-article/icse/2025/056900a051/215aWoRvPCE)]
    <details><summary><b>Abstract</b></summary>
    Large language models have catalyzed an unprecedented wave in code generation. While achieving significant advances, they blur the distinctions between machine- and human-authored source code, causing integrity and authenticity issues of software artifacts. Previous methods such as DetectGPT have proven effective in discerning machine-generated texts, but they do not identify and harness the unique patterns of machine-generated code. Thus, its applicability falters when applied to code. In this paper, we carefully study the specific patterns that characterize machine- and human-authored code. Through a rigorous analysis of code attributes such as lexical diversity, conciseness, and naturalness, we expose unique patterns inherent to each source. We particularly notice that the syntactic segmentation of code is a critical factor in identifying its provenance. Based on our findings, we propose DetectCodeGPT, a novel method for detecting machine-generated code, which improves DetectGPT by capturing the distinct stylized patterns of code. Diverging from conventional techniques that depend on external LLMs for perturbations, DetectCodeGPT perturbs the code corpus by strategically inserting spaces and newlines, ensuring both efficacy and efficiency. Experiment results show that our approach significantly outperforms state-of-the-art techniques in detecting machine-generated code.
    </details>

1. **Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education** (IEEE/ACM 2024) ![](https://img.shields.io/badge/Analysis-green)<br />
    *PAN, Wei Hung, CHOK, Ming Jie, WONG, Jonathan Leong Shan, Shin, Yung Xin, Poon, Yeong Shian*
    [[paper](https://ieeexplore.ieee.org/document/10554754/)]
    <details><summary><b>Abstract</b></summary>
    Educators are increasingly concerned about the usage of Large Language Models (LLMs) such as ChatGPT in programming education, particularly regarding the potential exploitation of imperfections in Artificial Intelligence Generated Content (AIGC) Detectors for academic misconduct. In this paper, we present an empirical study where the LLM is examined for its attempts to bypass detection by AIGC Detectors. This is achieved by generating code in response to a given question using different variants. We collected a dataset comprising 5,069 samples, with each sample consisting of a textual description of a coding problem and its corresponding human-written Python solution codes. These samples were obtained from various sources, including 80 from Quescol, 3,264 from Kaggle, and 1,725 from Leet-Code. From the dataset, we created 13 sets of code problem variant prompts, which were used to instruct ChatGPT to generate the outputs. Subsequently, we assessed the performance of five AIGC detectors. Our results demonstrate that existing AIGC Detectors perform poorly in distinguishing between human-written code and AI-generated code.
    </details>

1. **Is This You, LLM? Recognizing AI-written Programs with Multilingual Code Stylometry** (arXiv Dec 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) ![](https://img.shields.io/badge/Dataset-blue)<br />
    *Andrea Gurioli, Maurizio Gabbrielli, Maurizio Gabbrielli*
    [[paper](https://arxiv.org/abs/2412.14611)]
    <details><summary><b>Abstract</b></summary>
    With the increasing popularity of LLM-based code completers, like GitHub Copilot, the interest in automatically detecting AI-generated code is also increasing-in particular in contexts where the use of LLMs to program is forbidden by policy due to security, intellectual property, or ethical this http URL introduce a novel technique for AI code stylometry, i.e., the ability to distinguish code generated by LLMs from code written by humans, based on a transformer-based encoder classifier. Differently from previous work, our classifier is capable of detecting AI-written code across 10 different programming languages with a single machine learning model, maintaining high average accuracy across all languages (84.1% ± 3.8%).Together with the classifier we also release H-AIRosettaMP, a novel open dataset for AI code stylometry tasks, consisting of 121 247 code snippets in 10 popular programming languages, labeled as either human-written or AI-generated. The experimental pipeline (dataset, training code, resulting models) is the first fully reproducible one for the AI code stylometry task. Most notably our experiments rely only on open LLMs, rather than on proprietary/closed ones like ChatGPT.
    </details>

1. **EX-CODE: A Robust and Explainable Model to Detect AI-Generated Code** (Informationc 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Luana Bulla, Alessandro Midolo, Misael Mongiovì, Emiliano Tramontana*
    [[paper](https://www.mdpi.com/2078-2489/15/12/819)]
    <details><summary><b>Abstract</b></summary>
    Distinguishing whether some code portions were implemented by humans or generated by a tool based on artificial intelligence has become hard. However, such a classification would be important as it could point developers towards some further validation for the produced code. Additionally, it holds significant importance in security, legal contexts, and educational settings, where upholding academic integrity is of utmost importance. We present EX-CODE, a novel and explainable model that leverages the probability of the occurrence of some tokens, within a code snippet, estimated according to a language model, to distinguish human-written from AI-generated code. EX-CODE has been evaluated on a heterogeneous real-world dataset and stands out for its ability to provide human-understandable explanations of its outcomes. It achieves this by uncovering the features that for a snippet of code make it classified as human-written code (or AI-generated code).
    </details>

1. **Distinguishing LLM-generated from Human-written Code by Contrastive Learning** (arXiv Nov 2024) ![](https://img.shields.io/badge/Analysis-green) <br />
    *Xiaodan Xu, Chao Ni, Xinrong Guo, Shaoxuan Liu, Xiaoya Wang, Kui Liu, Xiaohu Yang*
    [[paper](http://arxiv.org/abs/2411.04704)]
    <details><summary><b>Abstract</b></summary>
    Large language models (LLMs), such as ChatGPT released by OpenAI, have attracted significant attention from both industry and academia due to their demonstrated ability to generate high-quality content for various tasks. Despite the impressive capabilities of LLMs, there are growing concerns regarding their potential risks in various fields, such as news, education, and software engineering. Recently, several commercial and open-source LLM-generated content detectors have been proposed, which, however, are primarily designed for detecting natural language content without considering the specific characteristics of program code. This paper aims to fill this gap by proposing a novel ChatGPT-generated code detector, CodeGPTSensor, based on a contrastive learning framework and a semantic encoder built with UniXcoder. To assess the effectiveness of CodeGPTSensor on differentiating ChatGPT-generated code from human-written code, we first curate a large-scale Human and Machine comparison Corpus (HMCorp), which includes 550K pairs of human-written and ChatGPT-generated code (i.e., 288K Python code pairs and 222K Java code pairs). Based on the HMCorp dataset, our qualitative and quantitative analysis of the characteristics of ChatGPT-generated code reveals the challenge and opportunity of distinguishing ChatGPT-generated code from human-written code with their representative features. Our experimental results indicate that CodeGPTSensor can effectively identify ChatGPT-generated code, outperforming all selected baselines.
    </details>

1. **An Empirical Study on Automatically Detecting AI-Generated Source Code: How Far Are We?** (arXiv Nov 2024) ![](https://img.shields.io/badge/Analysis-green) <br />
    *Hyunjae Suh, Mahan Tafreshipour, Jiawei Li, Adithya Bhattiprolu, Iftekhar Ahmed*
    [[paper](http://arxiv.org/abs/2411.04299)]
    <details><summary><b>Abstract</b></summary>
    Artificial Intelligence (AI) techniques, especially Large Language Models (LLMs), have started gaining popularity among researchers and software developers for generating source code. However, LLMs have been shown to generate code with quality issues and also incurred copyright/licensing infringements. Therefore, detecting whether a piece of source code is written by humans or AI has become necessary. This study first presents an empirical analysis to investigate the effectiveness of the existing AI detection tools in detecting AI-generated code. The results show that they all perform poorly and lack sufficient generalizability to be practically deployed. Then, to improve the performance of AI-generated code detection, we propose a range of approaches, including fine-tuning the LLMs and machine learning-based classification with static code metrics or code embedding generated from Abstract Syntax Tree (AST). Our best model outperforms state-of-the-art AI-generated code detector (GPTSniffer) and achieves an F1 score of 82.55. We also conduct an ablation study on our best-performing model to investigate the impact of different source code features on its performance.
    </details>

1. **CODEIP: A Grammar-Guided Multi-Bit Watermark for Large Language Models of Code** (arXiv Sept 2024) ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Batu Guan, Yao Wan, Zhangqian Bi, Zheng Wang, Hongyu Zhang, Pan Zhou, Lichao Sun*
    [[paper](http://arxiv.org/abs/2404.15639)]
    <details><summary><b>Abstract</b></summary>
    Large Language Models (LLMs) have achieved remarkable progress in code generation. It now becomes crucial to identify whether the code is AI-generated and to determine the specific model used, particularly for purposes such as protecting Intellectual Property (IP) in industry and preventing cheating in programming exercises. To this end, several attempts have been made to insert watermarks into machinegenerated code. However, existing approaches are limited to inserting only a single bit of information or overly depending on particular code patterns. In this paper, we introduce CODEIP, a novel multi-bit watermarking technique that inserts additional information to preserve crucial provenance details, such as the vendor ID of an LLM, thereby safeguarding the IPs of LLMs in code generation. Furthermore, to ensure the syntactical correctness of the generated code, we propose constraining the sampling process for predicting the next token by training a type predictor. Experiments conducted on a real-world dataset across five programming languages demonstrate the effectiveness of CODEIP in watermarking LLMs for code generation while maintaining the syntactical correctness of code.
    </details>

1. **An Empirical Study to Evaluate AIGC Detectors on Code Content** (ICASE 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Dataset-blue)<br />
   *JianWang, Shangqing Liu, Xiaofei Xie, Yi Li*
    [[paper](https://dl.acm.org/doi/10.1145/3691620.3695468)]
    <details><summary><b>Abstract</b></summary>
   Artificial Intelligence Generated Content (AIGC) has garnered considerable attention for its impressive performance, with Large Language Models (LLMs), like ChatGPT, emerging as a leading AIGC model that produces high-quality responses across various applications, including software development and maintenance. Despite its potential, the misuse of LLMs, especially in security and safety-critical domains, such as academic integrity and answering questions on Stack Overflow, poses significant concerns. Numerous AIGC detectors have been developed and evaluated on natural language data. However, their performance on code-related content generated by LLMs remains unexplored. To fill this gap, in this paper, we present an empirical study evaluating existing AIGC detectors in the software domain. We select three state-of-the-art LLMs, i.e., GPT-3.5, WizardCoder and CodeLlama, for machine-content generation. We further created a comprehensive dataset including 2.23M samples comprising code-related content for each model, encompassing popular software activities like Q&A (150K), code summarization (1M), and code generation (1.1M). We evaluated thirteen AIGC detectors, comprising six commercial and seven open-source solutions, assessing their performance on this dataset. Our results indicate that AIGC detectors perform less on code-related data than natural language data. Fine-tuning can enhance detector performance, especially for content within the same domain; but generalization remains a challenge.
    </details>

1. **Detecting AI-Generated Code Assignments Using Perplexity of Large Language Models** (AAAI 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
   *Zhenyu Xu, Victor S. Sheng*
    [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/30361)]
    <details><summary><b>Abstract</b></summary>
    Large language models like ChatGPT can generate humanlike code, posing challenges for programming education as students may be tempted to misuse them on assignments. However, there are currently no robust detectors designed specifically to identify AI-generated code. This is an issue that needs to be addressed to maintain academic integrity while allowing proper utilization of language models. Previous work has explored different approaches to detect AIgenerated text, including watermarks, feature analysis, and fine-tuning language models. In this paper, we address the challenge of determining whether a student’s code assignment was generated by a language model. First, our proposed method identifies AI-generated code by leveraging targeted masking perturbation paired with comprehensive scoring. Rather than applying a random mask, areas of the code with higher perplexity are more intensely masked. Second, we utilize a fine-tuned CodeBERT to fill in the masked portions, producing subtle modified samples. Then, we integrate the overall perplexity, variation of code line perplexity, and burstiness into a unified score. In this scoring scheme, a higher rank for the original code suggests it’s more likely to be AI-generated. This approach stems from the observation that AI-generated codes typically have lower perplexity. Therefore, perturbations often exert minimal influence on them. Conversely, sections of human-composed codes that the model struggles to understand can see their perplexity reduced by such perturbations. Our method outperforms current open-source and commercial text detectors. Specifically, it improves detection of code submissions generated by OpenAI’s text-davinci-003, raising average AUC from 0.56 (GPTZero baseline) to 0.87 for our detector.
    </details>

1. **GPTSniffer: A CodeBERT-based classifier to detect source code written by ChatGPT** (Journal of Systems and Software 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Phuong T. Nguyen, Juri Di Rocco, Claudio Di Sipio, Riccardo Rubei, Davide Di Ruscio, Massimiliano Di Penta*
    [[paper](https://www.sciencedirect.com/science/article/pii/S0164121224001043?via%3Dihub)]
    <details><summary><b>Abstract</b></summary>
    Since its launch in November 2022, ChatGPT has gained popularity among users, especially programmers who use it to solve development issues. However, while offering a practical solution to programming problems, ChatGPT should be used primarily as a supporting tool (e.g., in software education) rather than as a replacement for humans. Thus, detecting automatically generated source code by ChatGPT is necessary, and tools for identifying AI-generated content need to be adapted to work effectively with code. This paper presents GPTSniffer– a novel approach to the detection of source code written by AI – built on top of CodeBERT. We conducted an empirical study to investigate the feasibility of automated identification of AI-generated code, and the factors that influence this ability. The results show that GPTSniffer can accurately classify whether code is human-written or AI-generated, outperforming two baselines, GPTZero and OpenAI Text Classifier. Also, the study shows how similar training data or a classification context with paired snippets helps boost the prediction. We conclude that GPTSniffer can be leveraged in different contexts, e.g., in software engineering education, where teachers use the tool to detect cheating and plagiarism, or in development, where AI-generated code may require peculiar quality assurance activities.
    </details>

1. **Beyond Dataset Watermarking: Model-Level Copyright Protection for Code Summarization Models** (arXiv Aug 2024) ![](https://img.shields.io/badge/Watermarking-cyan) ![](https://img.shields.io/badge/Code_Protection-purple) <br />
    *Jiale Zhang, Haoxuan Li, Di Wu, Xiaobing Sun, Qinghua Lu, Guodong Long*
    [[paper](https://arxiv.org/abs/2410.14102)]
    <details><summary><b>Abstract</b></summary>
    Code Summarization Model (CSM) has been widely used in code production, such as online and web programming for PHP and Javascript. CSMs are essential tools in code production, enhancing software development efficiency and driving innovation in automated code analysis. However, CSMs face risks of exploitation by unauthorized users, particularly in an online environment where CSMs can be easily shared and disseminated. To address these risks, digital watermarks offer a promising solution by embedding imperceptible signatures within the models to assert copyright ownership and track unauthorized usage. Traditional watermarking for CSM copyright protection faces two main challenges: 1) dataset watermarking methods require separate design of triggers and watermark features based on the characteristics of different programming languages, which not only increases the computation complexity but also leads to a lack of generalization, 2) existing watermarks based on code style transformation are easily identifiable by automated detection, demonstrating poor concealment. To tackle these issues, we propose ModMark , a novel model-level digital watermark embedding method. Specifically, by fine-tuning the tokenizer, ModMark achieves cross-language generalization while reducing the complexity of watermark design. Moreover, we employ code noise injection techniques to effectively prevent trigger detection. Experimental results show that our method can achieve 100% watermark verification rate across various programming languages' CSMs, and the concealment and effectiveness of ModMark can also be guaranteed.
    </details>
    
1. **ACW: Enhancing Traceability of AI-Generated Codes Based on Watermarking** (arXiv Aug 2024) ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Boquan Li, Mengdi Zhang, Peixin Zhang, Jun Sun, Xingmei Wang*
    [[paper](https://arxiv.org/abs/2402.07518)]
    <details><summary><b>Abstract</b></summary>
    With the development of large language models, multiple AIs have become available for code generation (such as ChatGPT and StarCoder) and are adopted widely. It is often desirable to know whether a piece of code is generated by AI, and furthermore, which AI is the author. For instance, if a certain version of AI is known to generate vulnerable codes, it is particularly important to know the creator. Watermarking is broadly considered a promising solution and is successfully applied for identifying AI-generated text. However, existing efforts on watermarking AI-generated codes are far from ideal, and pose more challenges than watermarking general text due to limited flexibility and encoding space. In this work, we propose ACW (AI Code Watermarking), a novel method for watermarking AI-generated codes. The key idea of ACW is to selectively apply a set of carefully-designed semantic-preserving, idempotent code transformations, whose presence (or absence) allows us to determine the existence of watermarks. It is efficient as it requires no training or fine-tuning and works in a black-box manner. Our experimental results show that ACW is effective (i.e., achieving high accuracy on detecting AI-generated codes and extracting watermarks) as well as resilient, significantly outperforming existing approaches.
    </details>

1. **Whodunit: Classifying Code as Human Authored or GPT-4 generated- A case study on CodeChef problems** (MSR 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange)  <br />
    *Oseremen Joy Idialu, Noble Saji Mathews, Rungroj Maipradit, Joanne M. Atlee, Mei Nagappan*
    [[paper](https://dl.acm.org/doi/10.1145/3643991.3644926)]
    <details><summary><b>Abstract</b></summary>
    Artificial intelligence (AI) assistants such as GitHub Copilot and ChatGPT, built on large language models like GPT-4, are revolutionizing how programming tasks are performed, raising questions about whether code is authored by generative AI models. Such questions are of particular interest to educators, who worry that these tools enable a new form of academic dishonesty, in which students submit AI-generated code as their work. Our research explores the viability of using code stylometry and machine learning to distinguish between GPT-4 generated and human-authored code. Our dataset comprises human-authored solutions from CodeChef and AI-authored solutions generated by GPT-4. Our classifier outperforms baselines, with an F1-score and AUC-ROC score of 0.91. A variant of our classifier that excludes gameable features (e.g., empty lines, whitespace) still performs well with an F1-score and AUC-ROC score of 0.89. We also evaluated our classifier on the difficulty of the programming problem and found that there was almost no difference between easier and intermediate problems, and the classifier performed only slightly worse on harder problems. Our study shows that code stylometry is a promising approach for distinguishing between GPT-4 generated code and human-authored code.
    </details>

1. **Automatic Detection of LLM-generated Code: A Case Study of Claude 3 Haiku** (arXiv Sept 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Musfiqur Rahman, SayedHassan Khatoonabadi, Ahmad Abdellatif, Emad Shihab*
    [[paper](https://arxiv.org/abs/2409.01382)]
    <details><summary><b>Abstract</b></summary>
   Using Large Language Models (LLMs) has gained popularity among software developers for generating source code. However, the use of LLM-generated code can introduce risks of adding suboptimal, defective, and vulnerable code. This makes it necessary to devise methods for the accurate detection of LLM-generated code. Toward this goal, we perform a case study of Claude 3 Haiku (or Claude 3 for brevity) on CodeSearchNet dataset. We divide our analyses into two parts: function-level and class-level. We extract 22 software metric features, such as Code Lines and Cyclomatic Complexity, for each level of granularity. We then analyze code snippets generated by Claude 3 and their human-authored counterparts using the extracted features to understand how unique the code generated by Claude 3 is. In the following step, we use the unique characteristics of Claude 3-generated code to build Machine Learning (ML) models and identify which features of the code snippets make them more detectable by ML models. Our results indicate that Claude 3 tends to generate longer functions, but shorter classes than humans, and this characteristic can be used to detect Claude 3-generated code with ML models with 82% and 66% accuracies for function-level and class-level snippets, respectively.
    </details>

1. **Uncovering LLM-Generated Code: A Zero-Shot Synthetic Code Detector via Code Rewriting** (arXiv May 2024) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *MTong Ye, Yangkai Du, Tengfei Ma, Lingfei Wu, Xuhong Zhang, Shouling Ji, Wenhai Wang*
    [[paper](https://arxiv.org/abs/2405.16133)]
    <details><summary><b>Abstract</b></summary>
     Large Language Models (LLMs) have exhibited remarkable proficiency in generating code. However, the misuse of LLM-generated (Synthetic) code has prompted concerns within both educational and industrial domains, highlighting the imperative need for the development of synthetic code detectors. Existing methods for detecting LLM-generated content are primarily tailored for general text and often struggle with code content due to the distinct grammatical structure of programming languages and massive "low-entropy" tokens. Building upon this, our work proposes a novel zero-shot synthetic code detector based on the similarity between the code and its rewritten variants. Our method relies on the intuition that the differences between the LLM-rewritten and original codes tend to be smaller when the original code is synthetic. We utilize self-supervised contrastive learning to train a code similarity model and assess our approach on two synthetic code detection benchmarks. Our results demonstrate a notable enhancement over existing synthetic content detectors designed for general texts, with an improvement of 20.5% in the APPS benchmark and 29.1% in the MBPP benchmark.
    </details>

1. **MAGECODE: Machine-Generated Code Detection Method Using Large Language Models** (IEEE Access 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) ![](https://img.shields.io/badge/Dataset-blue) <br />
    *Hing Pham, Huyen Ha, Van Tong, Dung Hoang, Duc Tran, Tuyen Ngoc LE*
    [[paper](https://ieeexplore.ieee.org/document/10772217/?arnumber=10772217)]
    <details><summary><b>Abstract</b></summary>
    The widespread use of virtual assistants (e.g., GPT4 and Gemini, etc.) by students in their academic assignments raises concerns about academic integrity. Consequently, various machine-generated text (MGT) detection methods, developed from metric-based and model-based approaches, were proposed and shown to be highly effective. The model-based MGT methods often encounter difficulties when dealing with source codes due to disparities in semantics compared to natural languages. Meanwhile, the efficacy of metric-based MGT methods on source codes has not been investigated. Moreover, the challenge of identifying machine-generated codes (MGC) has received less attention, and existing solutions demonstrate low accuracy and high false positive rates across diverse human-written codes. In this paper, we take into account both semantic features extracted from Large Language Models (LLMs) and the applicability of metrics (e.g., Log-Likelihood, Rank, Log-rank, etc.) for source code analysis. Concretely, we propose MageCode, a novel method for identifying machine-generated codes. MageCode utilizes the pre-trained model CodeT5+ to extract semantic features from source code inputs and incorporates metric-based techniques to enhance accuracy. In order to assess the proposed method, we introduce a new dataset comprising more than 45,000 code solutions generated by LLMs for programming problems. The solutions for these programming problems which were obtained from three advanced LLMs (GPT4, Gemini, and Code-bison-32k), were written in Python, Java, and C++. The evaluation of MageCode on this dataset demonstrates superior performance compared to existing baselines, achieving up to 98.46% accuracy while maintaining a low false positive rate of less than 1%.
    </details>

1. **Program Code Generation with Generative AIs** (AI 2024) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Dataset-blue) <br />
    *Baskhad Idrisov, Tim Schlippe*
    [[paper](https://www.mdpi.com/1999-4893/17/2/62)]
    <details><summary><b>Abstract</b></summary>
    Our paper compares the correctness, efficiency, and maintainability of human-generated and AI-generated program code. For that, we analyzed the computational resources of AI- and human-generated program code using metrics such as time and space complexity as well as runtime and memory usage. Additionally, we evaluated the maintainability using metrics such as lines of code, cyclomatic complexity, Halstead complexity and maintainability index. For our experiments, we had generative AIs produce program code in Java, Python, and C++ that solves problems defined on the competition coding website leetcode.com. We selected six LeetCode problems of varying difficulty, resulting in 18 program codes generated by each generative AI. GitHub Copilot, powered by Codex (GPT-3.0), performed best, solving 9 of the 18 problems (50.0%), whereas CodeWhisperer did not solve a single problem. BingAI Chat (GPT-4.0) generated correct program code for seven problems (38.9%), ChatGPT (GPT-3.5) and Code Llama (Llama 2) for four problems (22.2%) and StarCoder and InstructCodeT5+ for only one problem (5.6%). Surprisingly, although ChatGPT generated only four correct program codes, it was the only generative AI capable of providing a correct solution to a coding problem of difficulty level hard. In summary, 26 AI-generated codes (20.6%) solve the respective problem. For 11 AI-generated incorrect codes (8.7%), only minimal modifications to the program code are necessary to solve the problem, which results in time savings between 8.9% and even 71.3% in comparison to programming the program code from scratch.
    </details>

1. **ChatGPT Code Detection: Techniques for Uncovering the Source of Code** (Algorithms 2024) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Marc Oedingen, Raphael C. Engelhardt, Robin Denz, Maximilian Hammer, Wolfgang Konen*
    [[paper](https://www.mdpi.com/2673-2688/5/3/53)]
    <details><summary><b>Abstract</b></summary>
    In recent times, large language models (LLMs) have made significant strides in generating computer code, blurring the lines between code created by humans and code produced by artificial intelligence (AI). As these technologies evolve rapidly, it is crucial to explore how they influence code generation, especially given the risk of misuse in areas such as higher education. The present paper explores this issue by using advanced classification techniques to differentiate between code written by humans and code generated by ChatGPT, a type of LLM. We employ a new approach that combines powerful embedding features (black-box) with supervised learning algorithms including Deep Neural Networks, Random Forests, and Extreme Gradient Boosting to achieve this differentiation with an impressive accuracy of 98%. For the successful combinations, we also examine their model calibration, showing that some of the models are extremely well calibrated. Additionally, we present white-box features and an interpretable Bayes classifier to elucidate critical differences between the code sources, enhancing the explainability and transparency of our approach. Both approaches work well, but provide at most 85–88% accuracy. Tests on a small sample of untrained humans suggest that humans do not solve the task much better than random guessing. This study is crucial in understanding and mitigating the potential risks associated with using AI in code generation, particularly in the context of higher education, software development, and competitive programming.
    </details>

1. **MCGMark: An Encodable and Robust Online Watermark for LLM-Generated Malicious Code** (ArXiv 2024) ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Kaiwen Ning, Jiachi Chen, Qingyuan Zhong, Tao Zhang, Yanlin Wang, Wei Li, Yu Zhang, Weizhe Zhang, Zibin Zheng*
    [[paper](https://arxiv.org/abs/2408.01354)]
    <details><summary><b>Abstract</b></summary>
    With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.
    </details>

1. **Is Watermarking LLM-Generated Code Robust?** () ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Tarun Suresh, Shubham Ugare, Gagandeep Singh, Sasa Misailovic*
    [[paper](https://arxiv.org/abs/2403.17983)]
    <details><summary><b>Abstract</b></summary>
    We present the first study of the robustness of existing watermarking techniques on Python code generated by large language models. Although existing works showed that watermarking can be robust for natural language, we show that it is easy to remove these watermarks on code by semantic-preserving transformations.
    </details>

1. **Discriminating Human-authored from ChatGPT-Generated Code Via Discernable Feature Analysis** (ISSREW 2023) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Dataset-blue) <br />
    *Ke Li, Sheng Hong, Cai Fu, Yunhe Zhang, Ming Liu*
    [[paper](https://ieeexplore.ieee.org/document/10301301)]
    <details><summary><b>Abstract</b></summary>
    The ubiquitous adoption of Large Language Generation Models (LLMs) in programming has highlighted the importance of distinguishing between human-written code and code generated by intelligent models. This paper specifically aims to distinguish ChatGPT-generated code from human-generated code. Our investigation reveals differences in programming style, technical level and readability between these two sources. Consequently, we develop a discriminative feature set for differentiation and evaluate its effectiveness through ablation experiments. In addition, we develop a dataset cleaning technique using temporal and spatial segmentation to mitigate dataset scarcity and ensure high quality, uncontaminated datasets. To further enrich the data resources, we apply "code transformation", "feature transformation" and "feature adaptation" techniques, generating a rich dataset of 100,000 lines of ChatGPT-generated code. The main contributions of our research include: proposing a discriminative feature set that yields high accuracy in distinguishing ChatGPT-generated code from human-authored code in binary classification tasks; devising methods for generating rich ChatGPT-generated code; and introducing a dataset cleansing strategy that extracts pristine, high-quality code datasets from open-source repositories, thereby achieving exceptional accuracy in code authorship attribution tasks.
    </details>

1. **Who Wrote this Code? Watermarking for Code Generation** (arXiv Jul 2023) ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Taehyun Lee, Seokhee Hong, Jaewoo Ahn, Ilgee Hong, Hwaran Lee, Sangdoo Yun, Jamin Shin, Gunhee Kim*
    [[paper](https://arxiv.org/abs/2305.15060)]
    <details><summary><b>Abstract</b></summary>
    Since the remarkable generation performance of large language models raised ethical and legal concerns, approaches to detect machine-generated text by embedding watermarks are being developed. However, we discover that the existing works fail to function appropriately in code generation tasks due to the task's nature of having low entropy. Extending a logit-modifying watermark method, we propose Selective WatErmarking via Entropy Thresholding (SWEET), which enhances detection ability and mitigates code quality degeneration by removing low-entropy segments at generating and detecting watermarks. Our experiments show that SWEET significantly improves code quality preservation while outperforming all baselines, including post-hoc detection methods, in detecting machine-generated code text.
    </details>

1. **CodeMark: Imperceptible Watermarking for Code Datasets against Neural Code Completion Models** (arXiv Jul 2023) ![](https://img.shields.io/badge/Watermarking-cyan) ![](https://img.shields.io/badge/Code_Protection-purple) <br />
    *Zhensu Sun, Xiaoning Du, Fu Song, Li Li*
    [[paper](https://dl.acm.org/doi/10.1145/3611643.3616297)]
    <details><summary><b>Abstract</b></summary>
    Code datasets are of immense value for training neural-networkbased code completion models, where companies or organizations have made substantial investments to establish and process these datasets. Unluckily, these datasets, either built for proprietary or public usage, face the high risk of unauthorized exploits, resulting from data leakages, license violations, etc. Even worse, the “black-box” nature of neural models sets a high barrier for externals to audit their training datasets, which further connives these unauthorized usages. Currently, watermarking methods have been proposed to prohibit inappropriate usage of image and natural language datasets. However, due to domain specificity, they are not directly applicable to code datasets, leaving the copyright protection of this emerging and important field of code data still exposed to threats. To fill this gap, we propose a method, named CodeMark, to embed user-defined imperceptible watermarks into code datasets to trace their usage in training neural code completion models. CodeMark is based on adaptive semantic-preserving transformations, which preserve the exact functionality of the code data and keep the changes covert against rule-breakers. We implement CodeMark in a toolkit and conduct an extensive evaluation of code completion models. CodeMark is validated to fulfill all desired properties of practical watermarks, including harmlessness to model accuracy, verifiability, robustness, and imperceptibility.
    </details>

1. **Towards Tracing Code Provenance with Code Watermarking** (ESEC/FSE 2023) ![](https://img.shields.io/badge/Watermarking-cyan) <br />
    *Wei Li, Borui Yang, Yujie Sun, Suyu Chen, Ziyun Song, Liyao Xiang, Xinbing Wang, Chenghu Zhou*
    [[paper](https://arxiv.org/abs/2305.12461)]
    <details><summary><b>Abstract</b></summary>
    Recent advances in large language models have raised wide concern in generating abundant plausible source code without scrutiny, and thus tracing the provenance of code emerges as a critical issue. To solve the issue, we propose CodeMark, a watermarking system that hides bit strings into variables respecting the natural and operational semantics of the code. For naturalness, we novelly introduce a contextual watermarking scheme to generate watermarked variables more coherent in the context atop graph neural networks. Each variable is treated as a node on the graph and the node feature gathers neighborhood (context) information through learning. Watermarks embedded into the features are thus reflected not only by the variables but also by the local contexts. We further introduce a pretrained model on source code as a teacher to guide more natural variable generation. Throughout the embedding, the operational semantics are preserved as only variable names are altered. Beyond guaranteeing code-specific properties, CodeMark is superior in watermarking accuracy, capacity, and efficiency due to a more diversified pattern generated. Experimental results show CodeMark outperforms the SOTA watermarking systems with a better balance of the watermarking requirements.
    </details>

1. **Protecting Intellectual Property of Large Language Model-Based Code Generation APIs via Watermarks** (CCS 2023) ![](https://img.shields.io/badge/Watermarking-cyan) ![](https://img.shields.io/badge/Code_Protection-purple)  <br />
    *Zongjie Li, Chaozheng Wang, Shuai Wang, Cuiyun Gao*
    [[paper](https://dl.acm.org/doi/10.1145/3576915.3623120)]
    <details><summary><b>Abstract</b></summary>
    The rise of large language model-based code generation (LLCG) has enabled various commercial services and APIs. Training LLCG models is often expensive and time-consuming, and the training data are often large-scale and even inaccessible to the public. As a result, the risk of intellectual property (IP) theft over the LLCG models (e.g., via imitation attacks) has been a serious concern. In this paper, we propose the first watermark (WM) technique to protect LLCG APIs from remote imitation attacks. Our proposed technique is based on replacing tokens in an LLCG output with their “synonyms” available in the programming language. A WM is thus defined as the stealthily tweaked distribution among token synonyms in LLCG outputs. We design six WM schemes (instantiated into over 30 WM passes) which rely on conceptually distinct token synonyms available in programming languages. Moreover, to check the IP of a suspicious model (decide if it is stolen from our protected LLCG API), we propose a statistical tests-based procedure that can directly check a remote, suspicious LLCG API. We evaluate our WM technique on LLCG models fine-tuned from two popular large language models, CodeT5 and CodeBERT. The evaluation shows that our approach is effective in both WM injection and IP check. The inserted WMs do not undermine the usage of normal users (i.e., high fidelity) and incur negligible extra cost. Moreover, our injected WMs exhibit high stealthiness and robustness against powerful attackers; even if they know all WM schemes, they can hardly remove WMs w
    </details>

1. **Distinguishing AI- and Human-Generated Code: a Case Study** (CCS 2023) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange)) <br />
    *Sufiyan Bukhari, Benjamin Tan, Lorenzo De Carli*
    [[paper](https://dl.acm.org/doi/10.1145/3605770.3625215)]
    <details><summary><b>Abstract</b></summary>
    While the use of AI assistants for code generation has the potential to revolutionize the way software is produced, assistants may generate insecure code, either by accident or as a result of poisoning attacks. They may also inadvertently violate copyright laws by mimicking code protected by restrictive licenses. We argue for the importance of tracking the provenance of AIgenerated code in the software supply chain, so that adequate controls can be put in place to mitigate risks. For that, it is necessary to have techniques that can distinguish between human- and AIgenerate code, and we conduct a case study in regards to whether such techniques can reliably work. We evaluate the effectiveness of lexical and syntactic features for distinguishing AI- and humangenerated code on a standardized task. Results show accuracy up to 92%, suggesting that the problem deserves further investigation.
    </details>

1. **RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation** (ISSREW 2022) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Ke Li, Sheng Hong, Cai Fu, Yunhe Zhang, Ming Liu*
    [[paper](https://ieeexplore.ieee.org/document/10301301)]
    <details><summary><b>Abstract</b></summary>
    The ubiquitous adoption of Large Language Generation Models (LLMs) in programming has highlighted the importance of distinguishing between human-written code and code generated by intelligent models. This paper specifically aims to distinguish ChatGPT-generated code from human-generated code. Our investigation reveals differences in programming style, technical level and readability between these two sources. Consequently, we develop a discriminative feature set for differentiation and evaluate its effectiveness through ablation experiments. In addition, we develop a dataset cleaning technique using temporal and spatial segmentation to mitigate dataset scarcity and ensure high quality, uncontaminated datasets. To further enrich the data resources, we apply "code transformation", "feature transformation" and "feature adaptation" techniques, generating a rich dataset of 100,000 lines of ChatGPT-generated code. The main contributions of our research include: proposing a discriminative feature set that yields high accuracy in distinguishing ChatGPT-generated code from human-authored code in binary classification tasks; devising methods for generating rich ChatGPT-generated code; and introducing a dataset cleansing strategy that extracts pristine, high-quality code datasets from open-source repositories, thereby achieving exceptional accuracy in code authorship attribution tasks.
    </details>

1. **On the Naturalness of Auto-Generated Code —Can We Identify Auto-Generated Code Automatically?—** (ICPC 2018) ![](https://img.shields.io/badge/Analysis-green) ![](https://img.shields.io/badge/Machine_Learning-orange) <br />
    *Masayuki Doi, Yoshiki Higo, Ryo Arima, Kento Shimonaka, Shinji Kusumoto*
    [[paper](https://ieeexplore.ieee.org/document/8972985/)]
    <details><summary><b>Abstract</b></summary>
    Recently, a variety of studies have been conducted on source code analysis. If auto-generated code is included in the target source code, it is usually removed in a preprocessing phase because the presence of auto-generated code may have negative effects on source code analysis. A straightforward way to remove autogenerated code is searching special comments that are included in the files of auto-generated code. However, it becomes impossible to identify auto-generated code with the way if such special comments have disappeared for some reasons. It is obvious that it takes too much effort to see source files one by one manually. In this paper, we propose a new technique to identify auto-generated code by using the naturalness of auto-generated code. We used a golden set that includes thousands of hand-made source files and source files generated by four kinds of compiler-compilers. Through the evaluation with the dataset, we confirmed that our technique was able to identify auto-generated code with over 99% precision and recall for all the cases.
    </details>


<a id="resources"></a>
## 🧰 Resources

<a id="datasets"></a>
### 📊 Datasets

Many datasets related to the analysis of Human writting and AI generated code has been done. Below is a non-exhaustive list of these popular choices:

<table align="center">
  <tbody>
    <tr align="center"> <th>Paper</th> <th>Dataset</th> </tr>
    <tr>
      <td> Discriminating Human-authored from ChatGPT-Generated Code Via Discernable Feature Analysis </td>
      <td> <a href="https://github.com/LiKe-rm/Human-and-ChatGPT-Code-Dataset/tree/main">HUMAN AND CHATGPT CODE DATASET</a> </td>
    </tr>
    <tr>
        <td>MAGECODE: Machine-Generated Code Detection Method Using Large Language Models</td>
        <td> <a href="https://huggingface.co/datasets/HungPhamBKCS/magecode-dataset">MAGECODE Dataset</a> </td>
    </tr>
    <tr>
        <td>Program Code Generation with Generative AIs</td>
         <td> <a href="https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/tree/main">AI-Human-Generated-Program-Code-Dataset</a> </td>
    </tr>
      <tr>
          <td>An Empirical Study to Evaluate AIGC Detectors on Code Content</td>
          <td><a href="https://sites.google.com/view/nlccd">NCLD-CDD</a></td>
      </tr>
      <tr>
          <td>Is This You, LLM? Recognizing AI-written Programs with Multilingual Code Stylometry</td>
          <td><a href="https://huggingface.co/datasets/isThisYouLLM/H-AIRosettaMP">H-AIRosettaMP</a></td>
      </tr>
  </tbody>
</table>


<a id="tools"></a>
### 🛠️ Tools


<table align="center">
  <tbody>
    <tr align="center">
      <th>Paper</th>
      <th>Tools</th>
    </tr>
    <tr>
      <td valign="top">
          Who Wrote this Code? Watermarking for Code Generation
      </td>
      <td valign="top">
          <a href="https://github.com/hongcheki/sweet-watermark.">SWEET</a>
      </td>
    </tr>
      <tr>
          <td>CodeMark: Imperceptible Watermarking for Code Datasets against Neural Code Completion Models</td>
          <td> <a href="https://github.com/v587su/CodeMark">CodeMark</a></td>
      </tr>
      <tr>
          <td>Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers</td>
          <td> <a href="https://github.com/YerbaPage/DetectCodeGPT">DetectCodeGPT</a></td>
      </tr>
      <tr>
          <td>MCGMark: An Encodable and Robust Online Watermark for LLM-Generated Malicious Code</td>
          <td><a href="https://github.com/KevinHeiwa/MCGTM">MCGTM</a></td>
      </tr>
      <tr>
          <td>GPTSniffer: A CodeBERT-based classifier to detect source code written by ChatGPT</td>
          <td><a href="https://github.com/MDEGroup/GPTSniffer/tree/master">GPT Sniffer</a></td>
      </tr>
  </tbody>
</table>
