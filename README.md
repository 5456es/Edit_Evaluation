<div align="center">

<img src="figs/logo.png" width="180px">

**An Easy-to-use Knowledge Editing Framework for Large Language Models.**

![](https://img.shields.io/badge/version-v0.0.1-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/last_commit-May-blue)
![](https://img.shields.io/badge/PRs-Welcome-red)

---

<p align="center">
  <a href="#requirements">Installation</a> •
  <a href="#use-easyedit">QuickStart</a> •
    <a href="https://zjunlp.gitbook.io/easyedit">Doc</a> •
    <a href="https://arxiv.org/abs/2401.01286">Paper</a> •
    <a href="https://huggingface.co/spaces/zjunlp/EasyEdit">Demo</a> •
    <a href="https://huggingface.co/datasets/zjunlp/KnowEdit">Benchmark</a> •
  <a href="#contributors">Contributors</a> •
  <a href="https://github.com/zjunlp/EasyEdit/blob/main/tutorial.pdf">Slides</a> •
    <a href="https://youtu.be/Gm6T0QaaskU", target="_blank">Video</a> •
   <a href="https://twitter.com/_akhaliq/status/1742371655765164133", target="_blank">Featured By AK</a>
</p>
</div>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [🔔News](#🔔news)
- [Editing Demo](#editing-demo)
- [Knowledge Editing](#knowledge-editing)
  - [Task Definition](#task-definition)
    - [Knowledge insert](#knowledge-insert)
    - [Knowledge update](#knowledge-update)
    - [Knowledge erase](#knowledge-erase)
  - [Comparisons of the different technologies](#comparisons-of-different-technologies)
  - [Evaluation](#evaluation)
- [🌟Overview](#🌟overview)
    - [Current Implementation](#current-implementation)
    - [Tutorial notebook](#tutorial-notebook)
- [Requirements](#requirements)
    - [🔧Pip Installation](#🔧pip-installation)
    - [Editing GPU memory usage](#editing-gpu-memory-usage)
- [📌Use EasyEdit](#📌use-easyedit)
  - [BaseEditor](#baseeditor)
    - [Introduction by a Simple Example](#introduction-by-a-simple-example)
  - [Evaluation](#evaluation-1)
  - [Trainer](#trainer)
- [Use EasyEdit with KnowEdit](#Use-easyedit-with-KnowEdit)
  - [Dataset](#Dataset)
  - [Usage](#usage)
- [Editing Performance](#editing-performance)
- [Citation](#citation)
- [🎉Contributors](#🎉contributors)
    - [Other Related Projects](#other-related-projects)

## 🔔News
- **New Reports**

    |     *Report*        |    Topic    | PPT Resource |
    | :-----------------: | :---------: | :------------: |
    | **COLING2024 tutorial**| Knowledge Editing for Large Language Models| [Google Drive](https://drive.google.com/file/d/1vFzRYjnzkuZaNdjdIxQwWbEybCY7YqY9/view?usp=sharing)   |
    | VALSE2024 tutorial| Knowledge Mechanism and Editing for Large Language Models| [Google Drive](https://drive.google.com/file/d/19T-InKopH-VHKAtphy9M6H366dXnenQX/view?usp=sharing)   |
    | AAAI2024 tutorial | Knowledge Editing for Large Language Models    | [Google Drive](https://drive.google.com/file/d/1fkTbVeRJSWmU7fBDeNf1OhHEkLSofQde/view?usp=sharing)   |

- 2024-07-29, The EasyEdit has added a new model editing algorithm [EMMET](https://arxiv.org/abs/2403.14236), which generalizes ROME to the batch setting. This essentially allows making batched edits using the ROME loss function.

- 2024-07-23, We release a new paper: "[Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/abs/2407.15017)", which reviews how knowledge is acquired, utilized, and evolves in large language models. This survey may provide the fundamental mechanisms for precisely and efficiently manipulating (editing) knowledge in LLMs.

- 2024-06-04, 🎉🎉 [EasyEdit Paper](https://arxiv.org/abs/2308.07269) has been accepted by the **ACL 2024** System Demonstration Track.

- 2024-06-03, We released a paper titled **["WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models"](https://arxiv.org/abs/2405.14768)**, along with introducing **a new editing task: [Continuous Knowledge Editing](#continuous-knowledge-editing)** and correspondding **lifelong editing method** called [WISE](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md).

- 2024-04-24, EasyEdit announced support for the **ROME method for [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)**. Users are advised to update their transformers package to version 4.40.0.

- 2024-03-29, EasyEdit introduced **rollback support for GRACE**. For a detailed introduction, refer to the [EasyEdit documentation](#use-easyedit). Future updates will gradually include rollback support for other methods.

- 2024-03-22, A new paper titled **"[Detoxifying Large Language Models via Knowledge Editing](https://arxiv.org/abs/2403.14472)"** was released, along with a new dataset named [SafeEdit](https://huggingface.co/datasets/zjunlp/SafeEdit) and a new **detoxification method** called [DINM](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md).

- 2024-03-12, Another paper titled **"[Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/abs/2403.06259)"** was released, introducing a new dataset named [ConceptEdit](https://huggingface.co/datasets/zjunlp/ConceptEdit).

- 2024-03-01, EasyEdit added support for a new method called **FT-M**. This method involves training a specific MLP layer **using cross-entropy loss on the target answer and masking the original text**. It outperforms the **FT-L** implementation in [ROME](https://github.com/kmeng01/rome). The author of issue https://github.com/zjunlp/EasyEdit/issues/173  is thanked for their advice.

- 2024-02-27, EasyEdit added support for a new method called [InstructEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md), with technical details provided in the paper **"[InstructEdit: Instruction-based Knowledge Editing for Large Language Models](https://arxiv.org/abs/2402.16123)"**.
<!-- - **2024-02-20 The AAAI2024 tutorial "*Knowledge Editing for Large Language Models*" has been canceled since speakers cannot present in person, we make this ppt[[Github](https://github.com/zjunlp/KnowledgeEditingPapers/blob/main/AAAI2024%40Tutorial_Knowledge%20Editing%20for%20LLMs.pdf)] [[Google Drive](https://drive.google.com/file/d/1fkTbVeRJSWmU7fBDeNf1OhHEkLSofQde/view?usp=sharing)] [[Baidu Pan](https://pan.baidu.com/s/1oJYgaMnxWIBE4kIcJuMSKg?pwd=p9j5)] available to the community**. -->

<details>
<summary><b>Previous News</b></summary>
  
- **2024-02-09 The EasyEdit has added the support for the Dynamic LoRA model editing method [MELO'AAAI24](https://arxiv.org/abs/2312.11795).**
- **2024-02-06 We release a new paper: "[EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models](https://arxiv.org/abs/2402.03049)" with an HF demo [EasyInstruct](https://huggingface.co/spaces/zjunlp/EasyInstruct).**
- **2024-02-06 We release a preliminary tool [EasyDetect](https://github.com/OpenKG-ORG/EasyDetect) for LLM hallucination detection，with a [demo](http://easydetect.openkg.cn/)**.
- **2024-01-24 The EasyEdit has added the support for editing [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) (manually update transformers==4.34.0), we have also fixed some bugs in evaluating MEND (slightly influence the performance).**
- **2024-01-16 The EasyEdit has added the support for the precise model editing method [PMET'AAAI24](https://arxiv.org/abs/2308.08742).**
- **2024-01-03  We release a new paper:"[A Comprehensive Study of Knowledge Editing for Large Language Models](https://arxiv.org/abs/2401.01286)" with a new benchmark [KnowEdit](https://huggingface.co/datasets/zjunlp/KnowEdit)! KnowEdit is constructed by re-organizing and cleaning existing datasests including WikiBio, ZsRE, WikiData Counterfact, WikiData Recent, convsent, Sanitation with new train/val/test spliting. Special thanks to the builders and maintainers of the those datasets.We are looking forward to any comments or discussions on this topic :)**
- **2023-12-06 The EasyEdit has added the support for the lifelong model editing method [GRACE'NeurIPS24](https://arxiv.org/abs/2211.11031).**
- **2023-11-18 Our tutorial "Knowledge Editing for Large Language Models" has been accepted by COLING 2024.**
- **2023-10-25 Our tutorial "Knowledge Editing for Large Language Models" has been accepted by AAAI 2024.**
- **2023-10-24 The EasyEdit has added the support for efficient editing of [Baichuan2](https://github.com/baichuan-inc/Baichuan2), [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B), [InternLM](https://github.com/InternLM/InternLM), [Qwen](https://github.com/QwenLM/Qwen) and fixed several bugs for a better user experience.**
- **2023-10-14 We release the [MultimodalEditor](#multimodaleditor) based on the paper "[Can We Edit Multimodal Large Language Models?](https://arxiv.org/abs/2310.08475)".**
- **2023-10-13 We release the paper "[Can We Edit Multimodal Large Language Models?](https://arxiv.org/abs/2310.08475)" accepted by EMNLP 2023.**
- **2023-10-08 Our paper "[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)" has been accepted by EMNLP 2023.**
- **2023-10-07 The EasyEdit has added the support for editing models with multiple GPUs, using huggingface [`Accelerate`](https://github.com/zjunlp/EasyEdit/blob/main/hparams/ROME/llama-7b.yaml#L24).**
- **2023-9-21 The EasyEdit has added the support for Parameter-Efficient Fine-Tuning through AdaLoRA to inject knowledge into the LLM.**
- **2023-8-31 The EasyEdit has added the support for official fine-tuning API for gpt-3.5-turbo to customize ChatGPT for your editing cases.**
- **2023-8-15 We release the paper "[EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269)."**
- **2023-7-12 We release version 0.0.1, supporting several knowledge editing techniques for LLMs. EasyEdit helps to better align LLMs with changing needs and values of users.**
- **2023-5-22 We release the paper "[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)" and provide a paper list at [PaperList](https://github.com/zjunlp/KnowledgeEditingPapers).**
- **2023-3-25 The EasyEdit project has been launched and is under development.**



This repository is a subproject of [KnowLM](https://github.com/zjunlp/KnowLM).
</details>

<!-- **EasyEdit** is now publicly open-sourced, with a [demo video](https://www.youtube.com/watch?v=NaQRvSYuQMo) and long-term maintenance. -->

---

> A Comprehensive Study of Knowledge Editing for Large Language Models [[paper](https://arxiv.org/abs/2401.01286)][[benchmark](https://huggingface.co/datasets/zjunlp/KnowEdit)][[code](https://github.com/zjunlp/EasyEdit)] 

> AAAI 2024 Tutorial [[Google Drive]()] [[Baidu Pan]()]

> AACL 2023 Tutorial [[Google Drive](https://drive.google.com/file/d/1EW-cusC_llCM0wEshkIdYuYrvfBPCDRz/view?usp=sharing)] [[Baidu Pan](https://pan.baidu.com/s/1NupastGJUzcUIAjI64J1tw?pwd=i5an)]


## Editing Demo

There is a demonstration of editing. The GIF file is created by [Terminalizer](https://github.com/faressoft/terminalizer).

<img src="figs/demo_usage_new.gif" width="550" height="470" align=center>

## Knowledge Editing

<div align=center><img src="./figs/ke.png" width="100%" height="80%" /></div>



### Task Definition

Deployed models may still make unpredictable errors. For example, LLMs notoriously _hallucinate_, _perpetuate bias_, and _factually decay_, so we should be able to adjust specific behaviors of pre-trained models.

**Knowledge editing** aims to adjust base model's $(f_\theta)$ behavior on the particular edit descriptor $[x_e, y_e]$​​​ efficiently.

### Multi Setting

#### Single Knowledge Editing

Evaluating the performance of the model after a single edit. The model reloads the original weights (e.g. LoRA discards the adapter weights) after a single edit. You should set **`sequential_edit=False`**

$$\theta' \leftarrow \text{arg} \min\limits_{\theta} (\Vert f_\theta(x_e) - y_e \Vert)$$

#### Continuous Knowledge Editing

This requires **sequentially editing**, and evaluation is performed after all knowledge updates have been applied:

$$\theta' \leftarrow \text{arg} \min\limits_{\theta} \sum_{e=1}^{\Vert X_e \Vert} (\Vert f_\theta(x_e) - y_e \Vert)$$

It makes parameter adjustments for $(x_e, y_e)$, where $x_e \in X_e$ and $f_\theta'(x_e) = y_e$. Here, $X_e$​ represents the whole **edit set**. To enable continuous editing, you can set **`sequential_edit=True`**: [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/WISE.md) (for more details).

### Multi Scenario

<details><summary> <b> Factual Knowledge Editing </b> </summary>

##### Knowledge insert

- Inject knowledge that LLMs have not seen before. such as:
  - *How many times has Messi won the World Cup? 0* $\rightarrow$ **1**:


##### Knowledge update

- Update outdated knowledge. such as:
  - *The president of USA: Donald Trump* $\rightarrow$ **Joe Biden**:


##### Knowledge erase

- Erase sensitive information. such as:
  - *The phone number of someone is XXXX* $\rightarrow$ **__**

Without influencing the model behavior on unrelated samples, the ultimate goal is to create an edited model $(f_\theta')$​​.

</details>

<details><summary> <b> Safety Editing </b> </summary>
**Detoxifying LLM** strives to build a safe and trustworthy large language model (LLM). Knowledge editing focuses on specific areas for permanent adjustment without compromising overall performance. Then, detoxifying LLM via knowledge editing leverages a small amount of data, usually an instance, to correct the toxic behaviors of the LLM. The edited LLM can defend against various malicious inputs. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)
</details>

<details><summary> <b> MultiModal Model Editing </b> </summary>

Editing Task for *Image Captioning* and *Visual Question Answering*. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md)
</details>

<details><summary> <b> Personality Editing </b> </summary>

The proposed task takes the preliminary attempt to edit LLMs' personalities by editing their opinions on specific topics, given that an individual's opinions can reflect aspects of their personality traits. We draw upon the established [BIG FIVE theory](https://en.wikipedia.org/wiki/Big_Five_personality_traits) as a basis for constructing our dataset and assessing the LLMs' personality expressions. [README](https://github.com/zjunlp/EasyEdit/blob/main/examples/PersonalityEdit.md)

**Evaluation**

Logits-based

- **ES**: evaluating the editing success rate based on the logits of pre-generated text.
- **DD**: evaluating whether the model changes opinions on other topics based on the logits of pre-generated text.

Generation-based

- **Acc**: the accuracy of the generated text after editing the model on target personality.
- **TPEI**: measuring whether generated opinion text from the edited model leans more towards the target personality.
- **PAE**: utilizing GPT-4 to evaluate the personality traits in generated text.

While for assessing **Acc** and **TPEI**, you can download the trained classifier from [here](https://huggingface.co/shai-msy/per-classifier).

</details>


### Comparisons of different technologies

<div align=center><img src="./figs/comparison.png" width="60%" height="48%" /></div>

### Evaluation

The knowledge editing process generally impacts the predictions for a broad set of inputs **that are closely** associated with the edit example, called the **editing scope**.

A successful edit should adjust the model’s behavior within the editing scope while remaining unrelated inputs:

$$
f_{\theta_{e}}(x) = \begin{cases}
y_e & \text{if } x \in I(x_e,y_e) \\
f_{\theta}(x) & \text{if } x \in O(x_e, y_e) \end{cases}
$$

- `Reliability`: the success rate of editing with a given editing descriptor
- `Generalization`: the success rate of editing within the editing scope
- `Locality`: whether the model's output changes after editing for unrelated inputs
- `Portability`: the success rate of editing for reasoning/application(one hop, synonym, logical generalization)
- `Efficiency`: time and memory consumption


## 🌟Overview

EasyEdit is a Python package for edit Large Language Models (LLM) like `GPT-J`, `Llama`, `GPT-NEO`, `GPT2`, `T5`(support models from **1B** to **65B**), the objective of which is to alter the behavior of LLMs efficiently within a specific domain without negatively impacting performance across other inputs. It is designed to be easy to use and easy to extend.

<h3 align="center">
<img src="figs/FrameWork.png">
</h3>

- EasyEdit contains a unified framework for **Editor**, **Method** and **Evaluate**, respectively representing the editing scenario, editing technique, and evaluation method.

- Each Knowledge Editing scenario comprises of three components:

  - `Editor`: such as BaseEditor(**Factual Knowledge** and **Generation** Editor) for LM, MultiModalEditor(**MultiModal Knowledge**).
  - `Method`: the specific knowledge editing technique used(such as **ROME**, **MEND**, ..).
  - `Evaluate`: **Metrics** for evaluating knowledge editing performance.
    - `Reliability`, `Generalization`, `Locality`, `Portability`

- The current supported knowledge editing techniques are as follows:
  
  
  - Memory-based: [SERAC](https://github.com/eric-mitchell/serac), [IKE](https://github.com/Zce1112zslx/IKE), [GRACE](https://github.com/thartvigsen/grace), [MELO](https://github.com/ECNU-ICALK/MELO), [WISE](https://arxiv.org/abs/2405.14768)
  - Meta-learning: [MEND](https://github.com/eric-mitchell/mend), [InstructEdit](https://github.com/zjunlp/EasyEdit/blob/main/examples/InstructEdit.md), [MALMEN](https://github.com/ChenmienTan/malmen)
  - Locate-then-edit: [KN](https://github.com/Hunter-DDM/knowledge-neurons), [ROME](https://github.com/kmeng01/rome), [MEMIT](https://github.com/kmeng01/memit), [PMET](https://github.com/xpq-tech/PMET), [DINM](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md), [R-ROME](https://github.com/scalable-model-editing/rebuilding-rome), [EMMET](https://github.com/scalable-model-editing/unified-model-editing)
  - [FT-L](https://github.com/kmeng01/rome)
  > Note 1: Due to the limited compatibility of this toolkit, some knowledge editing methods including  [T-Patcher](https://github.com/ZeroYuHuang/Transformer-Patcher), [KE](https://github.com/nicola-decao/KnowledgeEditor), [CaliNet](https://github.com/dqxiu/CaliNet)
  > are not supported. 
  >
  > Note 2: Similarly, the [MALMEN](https://github.com/ChenmienTan/malmen) method is only partially supported due to the same reasons and will continue to be improved.
#### Current Implementation

You can choose different editing methods according to your specific needs.
| **Method** | T5 | GPT-2 | GPT-J | GPT-NEO | LlaMA | Baichuan | ChatGLM | InternLM | Qwen | Mistral
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| FT | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AdaLoRA |  |  |  |  | ✅ |  | ✅ | | | |
| SERAC | ✅ | ✅ | ✅ | | ✅ |  | |  | | |
| IKE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |✅  | ✅ | ✅ | ✅ |
| MEND | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| KN   | ✅ | ✅ | ✅ |    | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROME | | ✅ | ✅ | ✅ | ✅ | ✅ |✅ | ✅ | ✅ | ✅ |
| r-ROME | | ✅ | ✅ | ✅ | ✅ | ✅ |✅ | ✅ | ✅ | ✅ |
| MEMIT | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅| ✅ | ✅ | ✅ |
| EMMET | | ✅ | ✅ |    | ✅ |    |   |    |    |    |
| GRACE | | ✅| ✅ |  |  ✅|  |  |  | | |
| MELO | |✅ |  |  |  |  |  |  | | |
| PMET | | | ✅ |  |  ✅|  |  |  | | |
| InstructEdit | | ✅ |  |  |  ✅|  |  |  | | |
| DINM| |✅ |  |  |✅  |  |  |  | | ✅|

<!-- |     KE       |  ✅  |  ✅  |  ✅  |  |  | -->



<!-- | **Method** | Model Name | Description |
| :--------: | :--------: | :--------: | 
| [FT-Api](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates) | [gpt-3.5-turbo(ChatGPT)](https://github.com/zjunlp/EasyEdit/blob/main/hparams/FT-Api/gpt-3.5-turbo.yaml) | official fine-tuing Api for gpt-3.5-turbo | -->

> ❗️❗️ If you intend to use Mistral, please update the `transformers` library to version 4.34.0 manually. You can use the following code: `pip install transformers==4.34.0`.

### Dataset

**Benchmark: KnowEdit** [[Hugging Face]](https://huggingface.co/datasets/zjunlp/KnowEdit)[[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit)[[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit)
> ❗️❗️ To be noted, **KnowEdit** is constructed by **re-organizing and extending** existing datasests including **WikiBio**, **ZsRE**, **WikiData<sub>Counterfact</sub>**,  **WikiData<sub>Recent</sub>**, **convsent**, **Sanitation** to make a comprehensive evaluation for knowledge editing. Special thanks to the builders and maintainers of the those datasets.

> Please note that Counterfact and WikiData<sub>Counterfact</sub> are not the same dataset.

<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Task</th>
    <th class="tg-7btt">Knowledge Insertion</th>
    <th class="tg-7btt" colspan="4">Knowledge Modification</th>
    <th class="tg-7btt">Knowledge Erasure</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Datasets</td>
    <td class="tg-c3ow">Wiki<sub>recent</sub></td>
    <td class="tg-c3ow">ZsRE</td>
    <td class="tg-c3ow">WikiBio</td>
    <td class="tg-c3ow"> WikiData<sub>counterfact</sub></td>
    <td class="tg-c3ow">Convsent</td>
    <td class="tg-c3ow">Sanitation</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Type</td>
    <td class="tg-c3ow">Fact</td>
    <td class="tg-c3ow">Question Answering</td>
    <td class="tg-c3ow">Hallucination</td>
    <td class="tg-c3ow">Counterfact</td>
    <td class="tg-c3ow">Sentiment</td>
    <td class="tg-c3ow">Unwanted Info</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Train</td>
    <td class="tg-c3ow">570</td>
    <td class="tg-c3ow">10,000</td>
    <td class="tg-c3ow">592</td>
    <td class="tg-c3ow">1,455</td>
    <td class="tg-c3ow">14,390</td>
    <td class="tg-c3ow">80</td>
  </tr>
  <tr>
    <td class="tg-c3ow"># Test</td>
    <td class="tg-c3ow">1,266</td>
    <td class="tg-c3ow">1301</td>
    <td class="tg-c3ow">1,392</td>
    <td class="tg-c3ow">885</td>
    <td class="tg-c3ow">800</td>
    <td class="tg-c3ow">80</td>
  </tr>
</tbody>
</table>

We provide **detailed scripts** for user to easily use KnowEdit, please refer to [examples](https://github.com/zjunlp/EasyEdit/blob/main/examples/KnowEdit.md).

<details><summary> <b> dataset description </b> </summary>

- ZsRE: is a context-free question-answering task. Given a question based on the subject and relation, the model is expected to provide the correct object as the answer. 
- Wiki<sub>recent</sub>: This dataset specifically focuses on triplets that have been recently inserted into WikiData after July 2022. 
- WikiBio: The original dataset was created by prompting GPT-3 to generate 238 Wikipedia-style biographies using subjects from the WikiBio.
- WikiData<sub>counterfact</sub>: Since tail entities are often not captured by models, and therefore are not suitable for testing modification edits, RippleEdit collects triplets about popular entities, where the subject corresponds to one of the top-viewed pages in Wikipedia.
- Convsent: This is a sentiment editing task that assesses the model's ability to modify a dialog agent's sentiment on a specific topic without affecting its responses to other topics.
- Sanitation: This dataset specifically addresses privacy concerns associated with learned language models. 
</details>


<details><summary> <b> dataset structure </b> </summary>

```text
knowedit
├── WikiBio
│   ├── wikibio-test-all.json
│   └── wikibio-train-all.json
├── ZsRE
│   └── ZsRE-test-all.json
├── wiki_counterfact
│   ├── test_cf.json
│   └── train_cf.json
├── convsent
│   ├── blender_test.json
│   ├── blender_train.json
│   └── blender_val.json
├── convsent
│   ├── trivia_qa_test.json
│   └── trivia_qa_train.json
└── wiki_recent
    ├── recent_test.json
    └── recent_train.json
```

</details>

---
#### Datasets for Chinese Knowledge: CKnowEdit

| **dataset** | HuggingFace| WiseModel | ModelScope | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |:--------------------------------------------------------------------------------: |
| CKnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/CKnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/CKnowEdit) | [[ModelScope]](https://modelscope.cn/datasets/ZJUNLP/CKnowEdit) | dataset for editing Chinese Knowledge |

- Here, you can follow [CKnowEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/CKnowEdit.md) to find more details about **CKnowEdit** and run Chinese knowledge editing experiments.

<details><summary> <b> dataset description </b> </summary>

**CKnowEdit** is a high-quality Chinese-language dataset for knowledge editing which is highly characterized by the Chinese language, with all data sourced from Chinese knowledge bases. It is meticulously designed to more deeply discern the nuances and challenges inherent in the comprehension of the Chinese language by current LLMs, providing a robust resource for refining Chinese-specific knowledge within LLMs.

The field descriptions for the data in **CKnowEdit** are as follows:

```python
"prompt": query inputed to the model (str)
"target_old": the incorrect response previously generated by the model (str)
"target_new": the accurate answer of the prompt (str)
"portability_prompt": new prompts related to the target knowledge (list or None)
"portability_answer": accurate answers corresponding to the portability_prompt (list or None)
"locality_prompt": new prompts unrelated to the target knowledge (list or None)
"locality_answer": accurate answers corresponding to the locality_prompt (list or None)
"rephrase": alternative ways to phrase the original prompt (list)
```
</details>

<details><summary> <b> dataset structure </b> </summary>

```text
CknowEdit
├── Chinese Literary Knowledge
│   ├── Ancient Poetry
│   ├── Proverbs
│   └── Idioms
├── Chinese Linguistic Knowledge
│   ├── Phonetic Notation
│   └── Classical Chinese
├── Chinese Geographical Knowledge
└── Ruozhiba
```
</details>

---

#### Datasets for Factual Knowledge
| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| _ZsRE_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Question Answering dataset using question rephrasings |
| _Counterfact_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Counterfact dataset using Entity replacement |


We provide zsre and counterfact datasets to verify the effectiveness of knowledge editing. You can download them here. [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing), [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky).

- For **locality**, in addition to testing unrelated instances, we also provide tests on distracting ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553)), other attribution, and other downstream tasks (such as commonsense reasoning).
- For **portability**, it tests whether the model can apply edited instances for inference. We provide evaluations for one-hop reasoning, subject alias, and inverse relation (eg, a one-to-one relationship between spouses should be bidirectionally edited).

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── counterfact
│   ├── counterfact-edit.json
│   ├── counterfact-train.json
│   └── counterfact-val.json
├── locality
│   ├── Commonsense Task
│   │   ├── piqa_valid-labels.lst
│   │   └── piqa_valid.jsonl
│   ├── Distracting Neighbor
│   │   └── counterfact_distracting_neighbor.json
│   └── Other Attribution
│       └── counterfact_other_attribution.json
├── portability
│   ├── Inverse Relation
│   │   └── zsre_inverse_relation.json
│   ├── One Hop
│   │   ├── counterfact_portability_gpt4.json
│   │   └── zsre_mend_eval_portability_gpt4.json
│   └── Subject Replace
│       ├── counterfact_subject_replace.json
│       └── zsre_subject_replace.json
└── zsre
    ├── zsre_mend_eval.json
    ├── zsre_mend_train_10000.json
    └── zsre_mend_train.json
```

- counterfact: original counterfact dataset using Entity replacement
- zsre: original question answering dataset using question rephrasings
- locality (evaluation for locality, see details in this [paper](https://arxiv.org/abs/2305.13172))
    - Commonsense Task: evaluation for other downstream tasks such as commonsense task
    - Distracting Neighbor: test on distracting neighborhood ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553))
    - Other Attribution
- portability
    - Inverse Relation: evaluation for one-to-one relationship such as `spouse`
    - One Hop: evaluation for one-hop reasoning
    - Subject Replace: evaluation for synonym replacement
    </details>

---

#### Datasets for Conceptual Knowledge: ConceptEdit

| **dataset** | Google Drive| HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| ConceptEdit | [[Google Drive]](https://drive.google.com/drive/folders/1Hp1DfIuj6Ih6ZLVENS-UmgJT8mRBlFC2?usp=drive_link) |[[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/ConceptEdit) | dataset for editing conceptual knowledge |

- Here, you can follow [ConceptEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/ConceptEdit.md) to run concept editing experiments.
  
<details><summary> <b> dataset description </b> </summary>

```text
data
└──concept_data.json
    ├──final_gpt2_inter.json
    ├──final_gpt2_intra.json
    ├──final_gptj_inter.json
    ├──final_gptj_intra.json
    ├──final_llama2chat_inter.json
    ├──final_llama2chat_intra.json
    ├──final_mistral_inter.json
    └──final_mistral_intra.json
```

**Concept Specific Evaluation Metrics**

- `Instance Change`: capturing the intricacies of these instance-level changes
- `Concept Consistency`: the semantic similarity of generated concept definition
</details>

---

#### Datasets for Multimodal Knowledge: MMEdit

| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| E-IC | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Image Captioning_ |
| E-VQA | [[Google Drive]](https://drive.google.com/drive/folders/1jBdTJxUb9wEeHnvG-RY8dv5_I4QlDpUS?usp=drive_link) | [[BaiduNetDisk]](https://pan.baidu.com/s/1g9nMv-5BJmztxYU-BWRdvg?pwd=ik5c) | dataset for editing _Visual Question Answering_ |

- All **images** used in **E-IC** and **E-VQA** are available for download at [Google Drive](https://drive.google.com/file/d/1fQzJBFkok5kFZT6QUuT-HCuYKk2Vb93O/view)
- For **locality**, it is the same as factual editing in order to measure whether unrelated facts retain their outputs.
- For **multimodal locality**, it assesses the impact of editing on the visual module, which is similar to regular **locality**.

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── caption
│   ├── caption_train_edit.json
│   └── caption_eval_edit.json
├── locality
│   ├── NQ dataset
│   │   ├── train.json
│   │   └── validation.json
├── multimodal_locality
│   ├── OK-VQA dataset
│   │   ├── okvqa_loc.json
└── vqa
    ├── vqa_train.json
    └── vqa_eval.json
```
- Multimodal locality (evaluation for multimodal locality, see dataset's details in this [paper](http://openaccess.thecvf.com/content\_CVPR\_2019/html/Marino\_OK-VQA\_A\_Visual\_Question\_Answering\_Benchmark\_Requiring\_External\_Knowledge\_CVPR\_2019\_paper.html)) 
</details>

---
#### Datasets for detoxifying LLMs: SafeEdit

| **dataset** | HuggingFace Dataset | Description |
| :--------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| SafeEdit |[[HuggingFace Dataset]](https://huggingface.co/datasets/zjunlp/SafeEdit) | dataset for detoxifying LLMs |

- Here, you can follow [SafeEdit.md](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md) to run detoxification editing experiments.
  
<details><summary> <b> dataset description </b> </summary>

```text
data
└──SafeEdit_train.json
└──SafeEdit_val.json
└──SafeEdit_test.json
    
```

**Detoxifying Specific Evaluation Metrics**
- `Defense Duccess (DS)`: the detoxification success rate of edited LLM for adversarial input (attack prompt + harmful question), which is used to modify LLM.
- `Defense Generalization (DG)`: the detoxification success rate of edited LLM for out-of-domain malicious inputs.
- `General Performance`: the side effects for unrelated task performance.
</details>

#### Tutorial notebook

| **Method** |          Description           |                                                 GPT-2                                                 |                                           LlaMA                                            |
| :--------: | :----------------------------: | :---------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
|   _IKE_    | In-Context Learning (ICL) Edit |       [[Colab-gpt2]](https://colab.research.google.com/drive/1m6Xg05XCs_WZKH0D9KJQqg9z0ZiDhEkL)       | [[Colab-llama]](https://colab.research.google.com/drive/1m6Xg05XCs_WZKH0D9KJQqg9z0ZiDhEkL) |
|   _ROME_   |    Locate-Then-Edit Neurons    | [[Colab-gpt2]](https://colab.research.google.com/drive/1KkyWqyV3BjXCWfdrrgbR-QS3AAokVZbr?usp=sharing) | [[Colab-llama]](https://colab.research.google.com/drive/1W18GPlBCV9K6lDy7eX8V5W0knTLr5r0A) |
|  _MEMIT_   |    Locate-Then-Edit Neurons    |       [[Colab-gpt2]](https://colab.research.google.com/drive/1P1lVklP8bTyh8uxxSuHnHwB91i-1LW6Z)       | [[Colab-llama]](https://colab.research.google.com/drive/19fKCKtVBU2fqj6eTvDokGoTrxvXkEPPq) |



## Requirements

#### 🔧Pip Installation

**Note: Please use Python 3.9+ for EasyEdit**
To get started, simply install conda and run:

```shell
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
...
pip install -r requirements.txt
```


#### Editing GPU memory usage
Our results are all based on the default configuration

|         | llama-2-7B | chatglm2 | gpt-j-6b | gpt-xl |
|:-------:|:----------:|:--------:|:--------:|:------:|
|   FT    |    60GB    |   58GB   |   55GB   |  7GB   |
|  SERAC  |    42GB    |   32GB   |   31GB   |  10GB  |
|   IKE   |    52GB    |   38GB   |   38GB   |  10GB  |
|  MEND   |    46GB    |   37GB   |   37GB   |  13GB  |
|   KN    |    42GB    |   39GB   |   40GB   |  12GB  |
|  ROME   |    31GB    |   29GB   |   27GB   |  10GB  |
|  MEMIT  |    33GB    |   31GB   |   31GB   |  11GB  |
| AdaLoRA |    29GB    |   24GB   |   25GB   |  8GB   |
|  GRACE  |    27GB    |          |   23GB   |  6GB   |
<!-- editing multimodal -->
## 📌Use EasyEdit

- Edit large language models(LLMs) around **_5 seconds_**

- Following example shows you how to perform editing with EasyEdit. More examples and tutorials can be found at [examples](https://github.com/zjunlp/EasyEdit/tree/main/examples)

### BaseEditor

> `BaseEditor`is the class for Language Modality Knowledge Editing. You can choose the appropriate editing method based on your specific needs.

- Due to different transformer versions and different GPU models, the editing results may fluctuate **slightly**.

#### Introduction by a Simple Example

With the modularity and flexibility of `EasyEdit`, you can easily use it to edit model.

**Step1: Define a PLM as the object to be edited.**
Choose the PLM to be edited. `EasyEdit` supports partial models(`T5`, `GPTJ`, `GPT-NEO`, `LlaMA` so far) retrievable on [HuggingFace](https://huggingface.co/). The corresponding configuration file directory is `hparams/YUOR_METHOD/YOUR_MODEL.YAML`, such as `hparams/MEND/gpt2-xl.yaml`, set the corresponding `model_name` to select the object for knowledge editing.

```yaml
model_name: gpt2-xl
model_class: GPT2LMHeadModel
tokenizer_class: GPT2Tokenizer
tokenizer_name: gpt2-xl
model_parallel: false # true for multi-GPU editing
```

**Step2: Choose the appropriate Knowledge Editing Method**

```python
## In this case, we use MEND method, so you should import `MENDHyperParams`
from easyeditor import MENDHyperParams
## Loading config from hparams/MEMIT/gpt2-xl.yaml
hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')
```

**Step3: Provide the edit descriptor and edit target**

```python
## edit descriptor: prompt that you want to edit
prompts = [
    'What university did Watts Humphrey attend?',
    'Which family does Ramalinaceae belong to',
    'What role does Denny Herzig play in football?'
]
## You can set `ground_truth` to None !!!(or set to original output)
ground_truth = ['Illinois Institute of Technology', 'Lecanorales', 'defender']
## edit target: expected output
target_new = ['University of Michigan', 'Lamiinae', 'winger']
```

**Step4: Combine them into a `BaseEditor`**
`EasyEdit` provides a simple and unified way to init `Editor`, like huggingface: **from_hparams**.

```python
## Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)
```

**Step5: Provide the data for evaluation**
Note that the data for portability and locality are both **optional**(set to None for basic editing success rate evaluation only). The data format for both is a **dict**, for each measurement dimension, you need to provide the corresponding prompt and its corresponding ground truth. Here is an example of the data:

```python
locality_inputs = {
    'neighborhood':{
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    },
    'distracting': {
        'prompt': ['Ray Charles, the violin Hauschka plays the instrument', 'Grant Hill is a professional soccer Magic Johnson is a professional', 'The law in Ikaalinen declares the language Swedish In Loviisa, the language spoken is'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}
```

In the above example, we evaluate the performance of the editing methods about "neighborhood" and "distracting".

**Step6: Edit and Evaluation**
Done! We can conduct Edit and Evaluation for your model to be edited. The `edit` function will return a series of metrics related to the editing process as well as the modified model weights. [`sequential_edit=True` for continuous editing]

```python
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    sequential_edit=False # True: start continuous editing ✈️
)
## metrics: edit success, rephrase success, locality e.g.
## edited_model: post-edit model
```
The maximum input length for EasyEdit is 512. If this length is exceeded, you will encounter the error "CUDA error: device-side assert triggered." You can modify the maximum length in the following file:[LINK](https://github.com/zjunlp/EasyEdit/blob/7d947abfa2975dcdbbd81a355b8f69d47e1b421f/easyeditor/evaluate/evaluate_utils.py#L115)

**Step7: RollBack**
In sequential editing, if you are not satisfied with the outcome of one of your edits and you do not wish to lose your previous edits, you can use the rollback feature to undo your previous edit. Currently, we only support the GRACE method. All you need to do is a single line of code, using the edit_key to revert your edit.

```
editor.rolllback('edit_key')
```
In EasyEdit, we default to using target_new as the edit_key
### Evaluation

We specify the return metrics as `dict` format, including model prediction evaluations before and after editing. For each edit, it will include the following metrics:

- `rewrite_acc` $\rightarrow$ **Reliablilty**
- `rephrase_acc` $\rightarrow$ **Generalization**
- `locality` $\rightarrow$ **Locality**
- `portablility` $\rightarrow$ **Portablility**

```json
{
    "post": {
        "rewrite_acc": ,
        "rephrase_acc": ,
        "locality": {
            "YOUR_LOCALITY_KEY": ,
            //...
        },
        "portablility": {
            "YOUR_PORTABILITY_KEY": ,
            //...
        },
    },
    "pre": {
        "rewrite_acc": ,
        "rephrase_acc": ,
        "portablility": {
            "YOUR_PORTABILITY_KEY": ,
            //...
        },
    }
}
```

- For evaluation for Reliablilty, you only need to provide the corresponding editing `prompts` and editing `target_new`.
- For evaluation for Generalization, `rephrase_prompts` are required.
- For evaluation for Locality and Portablility, you need to define the name of the corresponding metric, as well as `prompts` and `ground_truth`.
  - > Note: the length needs to be equal to the edit prompts

### Trainer

- meta-learning based: `MEND`
- memory-based routing: `SERAC`

For above editing methods, pre-training of corresponding meta-networks or classifiers is required. Therefore, in EasyEdit, we provide a unified framework for pretraining the relevant network structures. Take the training MEND for example:

- **Step 1** and **Step 2** are the same as the example above, which involves selecting the appropriate editing model and editing method.

**Step3: Provide the edit training set**
The currently supported and available datasets are: `zsre` and `counterfact`([Google Drive](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing)). Please place them in the "data" directory and initialize the dataset_class (`ZsreDataset` for zsre and `CounterFactDataset` for counterfact) to load the corresponding training set.

```python
train_ds = ZsreDataset('./data/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre_mend_eval.json', config=training_hparams)
```

**Step4: Combine them into a `Trainer`**

```python
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
```

**Step5: Run and Edit**
Done! We can conduct Run and Evaluation.

```python
trainer.run()
```

- Run: The `CHECKPOINT` will be saved to the path `results_dir`.
- Edit: Set the `archive` field in the **hparams file** to `CHECKPOINT`. EasyEdit will automatically load the corresponding pre-trained weights during the editing process([Go to edit](#use-easyedit)).

**Training Example**
```python
from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset

training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/llama-7b.yaml')
train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()
```

<!-- ## Overall Results
> Note that the following experimental results are from this [paper](https://arxiv.org/abs/2305.13172).The actual editing performance of this tool is still under testing and will be announced **as soon as possible**.
*  We tested the editing performance of different knowledge editing methods on various model, the test results are shown in the table below(`-` refers to the results that the methods empirically fail to edit LLMs). -->
<!--
- For `zsre` dataset:

<div style="text-align: center">
<table style="text-align: center">
    <tr>
        <th></th><th colspan="3" style="text-align: center;">T5-3B</th><th colspan="3" style="text-align: center;">GPT-J</th>
    </tr>
    <tr>
        <td><b>Method</b></td><td>Reliability</td><td>Generalization</td><td>Locality</td><td>Reliability</td><td>Generalization</td><td>Locality</td>
    </tr>
    <tr>
        <td>FT</td><td>20.71</td><td>19.68</td><td>89.01</td><td>54.70</td><td>49.20</td><td>37.24</td>
    </tr>
    <tr>
        <td>SERAC</td><td>99.80</td><td>99.66</td><td>98.13</td><td>90.16</td><td>89.96</td><td>99.90</td>
    </tr>
    <tr>
        <td>IKE</td><td>67.00</td><td>67.11</td><td>63.60</td><td>99.96</td><td>99.87</td><td>59.21</td>
    </tr>
    <tr>
        <td>KE</td><td>3.00</td><td>5.40</td><td>96.43</td><td>6.60</td><td>7.80</td><td>94.18</td>
    </tr>
    <tr>
        <td>MEND</td><td>78.80</td><td>89.80</td><td>98.45</td><td>45.60</td><td>48.00</td><td>88.21</td>
    </tr>
    <tr>
        <td>KN</td><td>22.51</td><td>22.70</td><td>16.43</td><td>11.34</td><td>9.40</td><td>90.03</td>
    </tr>
    <tr>
        <td>ROME</td><td>-</td><td>-</td><td>-</td><td>99.18</td><td>94.90</td><td>99.19</td>
    </tr>
    <tr>
        <td>MEMIT</td><td>-</td><td>-</td><td>-</td><td>99.23</td><td>87.16</td><td>99.62</td>
    </tr>
</table>
</div>

- For `counterfact` dataset:

<div style="text-align: center">
<table style="text-align: center">
    <tr>
        <th></th><th colspan="3" style="text-align: center;">T5-3B</th><th colspan="3" style="text-align: center;">GPT-J</th>
    </tr>
    <tr>
        <td><b>Method</b></td><td>Reliability</td><td>Generalization</td><td>Locality</td><td>Reliability</td><td>Generalization</td><td>Locality</td>
    </tr>
    <tr>
        <td>FT</td><td>33.57</td><td>23.54</td><td>72.72</td><td>99.90</td><td>97.53</td><td>1.02</td>
    </tr>
    <tr>
        <td>SERAC</td><td>99.89</td><td>98.71</td><td>99.93</td><td>99.78</td><td>99.41</td><td>98.89</td>
    </tr>
    <tr>
        <td>IKE</td><td>97.77</td><td>82.99</td><td>37.76</td><td>99.61</td><td>72.67</td><td>35.57</td>
    </tr>
    <tr>
        <td>KE</td><td>1.00</td><td>1.40</td><td>96.28</td><td>13.40</td><td>11.00</td><td>94.38</td>
    </tr>
    <tr>
        <td>MEND</td><td>81.40</td><td>93.40</td><td>91.58</td><td>73.80</td><td>74.20</td><td>93.75</td>
    </tr>
    <tr>
        <td>KN</td><td>47.86</td><td>46.78</td><td>57.10</td><td>1.66</td><td>1.38</td><td>58.28</td>
    </tr>
    <tr>
        <td>ROME</td><td>-</td><td>-</td><td>-</td><td>99.80</td><td>86.63</td><td>93.61</td>
    </tr>
    <tr>
        <td>MEMIT</td><td>-</td><td>-</td><td>-</td><td>99.90</td><td>73.13</td><td>97.17</td>
    </tr>
</table>
</div> -->

# Use EasyEdit with KnowEdit
## Dataset

KnowEdit is a benchmark dataset of knowledge editing for LLMs. You can easily obtain KnowEdit from HuggingFace, HuggingFace, and ModelScope.

| **dataset** | HuggingFace| HuggingFace | ModelScope |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| KnowEdit | [[HuggingFace]](https://huggingface.co/datasets/zjunlp/KnowEdit) | [[WiseModel]](https://wisemodel.cn/datasets/zjunlp/KnowEdit) | [[ModelScope]](https://www.modelscope.cn/datasets/zjunlp/KnowEdit) |


## Usage 

We provide detailed scripts for user to easily use KnowEdit, please refer to [examples](https://github.com/zjunlp/EasyEdit/blob/main/examples/KnowEdit.md).

# Editing Performance

We present editing results of the four metrics on [LlaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) using EasyEdit. We adopt [ZsRE](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) as the test dataset.

> ❗️❗️Editing `llama-2-7B` requires 40G+ VRAM on GPU. (OOM [solution](https://github.com/zjunlp/EasyEdit/issues/9#issuecomment-1687284658))

|       | Reliability | Generalization |  Locality  | Portability |
| :---: | :---------: | :------------: | :--------: | :---------: |
| FT  |    56.94    |     52.02      |   96.32    |    51.03    |
| SERAC |    99.49    |     99.13      | **100.00** |    57.82    |
|  IKE  | **100.00**  |   **99.98**    |   69.19    |  **67.56**  |
| MEND  |    94.24    |     90.27      |   97.04    |    56.95    |
|  KN   |    28.95    |     28.43      |   65.43    |    37.18    |
| ROME  |    92.45    |     87.04      |   99.63    |    57.47    |
| MEMIT |    92.94    |     85.97      |   99.49    |    60.64    |



We also present editing results of KnowEdit on [LlaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) using EasyEdit. 

| DataSet                  | Metric        | SERAC  | ICE    | AdaLoRA | MEND   | ROME   | MEMIT  | FT-L   | FT     |
|--------------------------|---------------|--------|--------|---------|--------|--------|--------|--------|--------|
| **WikiData_recent**      |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 98.68  | 60.74  | 65.61   | 76.88  | 85.08  | 85.32  | 71.18  | 31.24  |
|                          | Portability | 63.52  | 36.93  | 47.22   | 50.11  | 37.45  | 37.94  | 48.71  | 15.91  |
|                          | Locality     | 100.00 | 33.34  | 55.78   | 92.87  | 66.2   | 64.78  | 63.7   | 3.65   |
|                          | Fluency     | 553.19 | 531.01 | 537.51  | 586.34 | 574.28 | 566.66 | 549.35 | 428.67 |
| **ZsRE**                 |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.67  | 66.01  | 69.86   | 96.74  | 96.57  | 83.07  | 54.65  | 36.88  |
|                          | Portability | 56.48  | 63.94  | 52.95   | 60.41  | 52.20  | 51.43  | 45.02  | 8.72   |
|                          | Locality   | 30.23  | 23.14  | 72.21   | 92.79  | 27.14  | 25.46  | 71.12  | 0.31   |
|                          | Fluency     | 410.89 | 541.14 | 532.82  | 524.33 | 570.47 | 559.72 | 474.18 | 471.29 |
| **WikiBio**              |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.69  | 95.53  | 97.02   | 93.66  | 95.05  | 94.29  | 66.27  | 95.64  |
|                          | Locality    | 69.79  | 47.90  | 57.87   | 69.51  | 46.96  | 51.56  | 60.14  | 13.38  |
|                          | Fluency    | 606.95 | 632.92 | 615.86  | 609.39 | 617.25 | 616.65 | 604.00 | 589.22 |
| **WikiData_counterfact** |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 99.99  | 69.83  | 72.14   | 78.82  | 83.21  | 83.41  | 51.12  | 26.78  |
|                          | Portability | 76.07  | 45.32  | 55.17   | 57.53  | 38.69  | 40.09  | 39.07  | 16.94  |
|                          | Locality    | 98.96  | 32.38  | 66.78   | 94.16  | 65.4   | 63.68  | 62.51  | 0.29   |
|                          | Fluency     | 549.91 | 547.22 | 553.85  | 588.94 | 578.84 | 568.58 | 544.80 | 483.71 |
| **ConvSent**             |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 62.75  | 52.78  | 44.89   | 50.76  | 45.79  | 44.75  | 49.50  | 61.93  |
|                          | Locality    | 0.26   | 49.73  | 0.18    | 3.42   | 0.00   | 0.00   | 0.00   | 0.00   |
|                          | Fluency     | 458.21 | 621.45 | 606.42  | 379.43 | 606.32 | 602.62 | 607.86 | 546.24 |
| **Sanitation**           |               |        |        |         |        |        |        |        |        |
|                          | Edit Succ.  | 0.00   | 72.50  | 2.50    | 0.00   | 85.00  | 48.75  | 0.00   | 60.00  |
|                          | Locality    | 100.00 | 56.58  | 65.50   | 5.29   | 50.31  | 67.47  | 14.78  | 42.61  |
|                          | Fluency     | 416.29 | 794.15 | 330.44  | 407.18 | 465.12 | 466.10 | 439.10 | 351.39 |
> ❗️❗️ **Please note that if you wish to reproduce the results regarding Rome on Knowedi, ensure that `fp16: False`.**

> For the locality metric, we calculate the score based on the proportion of tokens that remain unchanged before and after editing. For example, if the output tokens before editing are [29, 234, 334] and after editing are [29, 234, 333], the locality score for this data would be 66.67. For the portability metric, we calculate it by taking the average of all sub-scores under the portability category.

<details><summary> <b> TO DO </b> </summary>
In next version, we plan to:

- Explore and integrate more robust editing methods, focusing on `locality` and `portability` metrics.
- Provide a comprehensive evaluation suite for editing methods, including fact modification, fact erasure and hallucination erasure.
- Provide a causal analysis component for analyzing knowledge storage mechanisms.
- knowledge editing for other tasks(except factual editing), like `personality editing`, etc.

Meanwhile, we will offer long-term maintenance to fix bugs, solve issues and meet new requests. So if you have any problems, please put issues to us.

</details>

## Citation

Please cite our paper if you use EasyEdit in your work.

```bibtex

@article{zhang2024comprehensive,
  title={A Comprehensive Study of Knowledge Editing for Large Language Models},
  author={Zhang, Ningyu and Yao, Yunzhi and Tian, Bozhong and Wang, Peng and Deng, Shumin and Wang, Mengru and Xi, Zekun and Mao, Shengyu and Zhang, Jintian and Ni, Yuansheng and others},
  journal={arXiv preprint arXiv:2401.01286},
  year={2024}
}

@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

@article{cheng2023edit,
  title={Can We Edit Multimodal Large Language Models?}, 
  author={Cheng, Siyuan and Tian, Bozhong and Liu, Qingbin and Chen, Xi and Wang, Yongheng and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2310.08475},
  year={2023}
}

@article{mao2023editing,
  title={Editing personality for llms},
  author={Mao, Shengyu and Zhang, Ningyu and Wang, Xiaohan and Wang, Mengru and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02168},
  year={2023}
}

@misc{knowlm,
  author = {Ningyu Zhang and Jintian Zhang and Xiaohan Wang and Honghao Gui and Kangwei Liu and Yinuo Jiang and Xiang Chen and Shengyu Mao and Shuofei Qiao and Yuqi Zhu and Zhen Bi and Jing Chen and Xiaozhuan Liang and Yixin Ou and Runnan Fang and Zekun Xi and Xin Xu and Lei Li and Peng Wang and Mengru Wang and Yunzhi Yao and Bozhong Tian and Yin Fang and Guozhou Zheng and Huajun Chen},
  title = {KnowLM Technical Report},
  year = {2023},
 url = {http://knowlm.zjukg.cn/},
}
```

## 🎉Contributors

<a href="https://github.com/zjunlp/EasyEdit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/EasyEdit" />
</a>

We thank all the contributors to this project, more contributors are welcome!

#### Other Related Projects

- [ROME](https://github.com/kmeng01/rome)
- [FastEdit](https://github.com/hiyouga/FastEdit)
- [GRACE](https://github.com/Thartvigsen/GRACE)
- [MELO](https://github.com/ECNU-ICALK/MELO)
- [PMET](https://github.com/xpq-tech/PMET)
- [VLKEB](https://github.com/VLKEB/VLKEB)
- [PitfallsKnowledgeEditing](https://github.com/zjunlp/PitfallsKnowledgeEditing)
- [BiasEdit](https://github.com/zjunlp/BiasEdit)
- [WikiLLM](https://github.com/laramohan/wikillm)
- [PEAK](https://github.com/mjy1111/PEAK)
- [Debugger](https://github.com/openai/transformer-debugger)
- [LTE](https://github.com/YJiangcm/LTE)
- [r-ROME](https://github.com/scalable-model-editing/rebuilding-rome)
- [dive-into-llms](https://github.com/Lordog/dive-into-llms)

🙌 We would like to express our heartfelt gratitude for the contribution of [FastEdit](https://github.com/hiyouga/FastEdit), [ROME](https://github.com/kmeng01/rome), [GRACE](https://github.com/Thartvigsen/GRACE), [MELO](https://github.com/ECNU-ICALK/MELO), [PMET](https://github.com/xpq-tech/PMET) to our project, as we have utilized portions of their source code in our project. Many thanks to all the colleagues in the community for submitting issues and providing technical support. Appreciation is also extended to all PR contributors, and issue feedback providers during the EasyEdit version iterations, especially ancelia06 for correcting the grammar of README.
