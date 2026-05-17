# Grasp Point Prediction
A Comparative Study of CLIP, Grounding DINO, and Florence-2 for Language-Guided Grasp-Relevant Point Prediction

## 🎯 Project Overview

This repository contains our project investigating **Computer Vision (CV)** by focusing on **Vision-Language Models (VLMs)** - AI systems that process both images and text - for the task of grasp-relevant 2D point prediction. 

Every year, millions of elderly and disabled individuals struggle to perform simple activities like picking up a glass of water or retrieving their medication. These are moments that caregivers must drop everything to address  which is costing the U.S. healthcare system approximately [$177.5 billion in 2026](https://www.mcknightsseniorliving.com/news/us-healthcare-spending-for-nfs-and-ccrcs-to-reach-337-4-billion-by-2032-cms-says/) and quietly eroding patients' sense of independence and dignity. If a robot could simply listen to a language instruction and act on it precisely the way a human assistant would, it could free caregivers and give people back their autonomy, safety, and quality of life. 

We aim to move beyond **coarse bounding-box localization** toward predicting precise (x, y) pixel coordinates that are meaningful for robotic interaction. **Our goal** is to achieve high-precision, language-guided grasp-relevant point prediction to improve lives, increase productivity, and discover other things we could think of as we complete this project. Toward this vision, our project takes a first step by asking a fundamental question: **Can language make a robot not just see an object, but understand how to interact with it the way a human would?** Our primary answer is integrating AI/ML into robotics, as these two fields of study are rapidly growing.

This project pursues three concrete objectives. First, we benchmark three vision-language architectures, a supervised CLIP-based regression head, and two zero-shot models (Grounding DINO and Florence-2) on the task of predicting precise (x, y) grasp-relevant pixel coordinates from RGB images alone. Second, we evaluate whether natural language prompt variation (e.g., "point to" versus "grasp the") causes statistically meaningful shifts in predicted interaction points, testing whether these models genuinely understand language or merely respond to visual patterns. Third, we establish a reproducible evaluation framework using pixel error, normalized error, and success rate across 48 object categories that future work can build on as models and datasets improve.

We draw on **GraspMoLMo** (Deshpande et al., 2025) as a motivating reference, as it demonstrates that natural language instructions can guide semantically meaningful grasp point selection (e.g., grasping a teapot handle vs. its body). While GraspMoLMo relies on RGB-D input and a large-scale synthetic dataset, our work investigates how much language-guided grasp localization is achievable from **RGB images alone**, using real-world annotations from PixMo-Points a deliberate simplification to isolate the role of visual-language grounding without depth sensing.

## 👥 Team Members (CRediT Statement)

* **Priyadarshini Rajmohan** - Conceptualization, Data Curation, Methodology, Formal Analysis, Investigation (CLIP + regressor), Visualization, Writing - Review & Editing.
* **Poojitha Alam** - Conceptualization, Methodology, Formal Analysis, Investigation (GroundingDINO), Visualization, Project Administration, Writing - Review & Editing.
* **Jannine G. D. MacGormain** - Conceptualization, Methodology, Formal Analysis, Investigation (Florence-2), Visualization, Project Administration, Writing - Original Draft

## 📁 Project Structure
```text
data/           ← pixmo_subset_v3 dataset
notebooks/      ← all model training and inference notebooks
outputs/        ← metrics and results
```

## Models Compared
- CLIP
- Grounding DINO  
- Florence-2

## 📁 Dataset
```text
Pixmo subset v3 — images with language-guided grasp point annotations
```

## 🏗️ Technical Environment & Dependencies

Our project's evaluation pipeline and comparative study were made possible by the following architectures, computational frameworks, and open-source libraries:

| Functional Domain | Technologies & Ecosystem |
| :--- | :--- |
| **Evaluated Architectures** | [![CLIP](https://img.shields.io/badge/CLIP-OpenAI-412991?logo=openai&logoColor=white)](https://github.com/openai/CLIP) [![Grounding DINO](https://img.shields.io/badge/Grounding_DINO-IDEA_Research-0052CC)](https://github.com/IDEA-Research/GroundingDINO) [![Florence-2](https://img.shields.io/badge/Florence--2-Microsoft-00A4EF?logo=microsoft&logoColor=white)](https://huggingface.co/microsoft/Florence-2-large) |
| **Deep Learning Frameworks** | [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/) |
| **Data Processing & Statistical Analysis** | [![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/) |
| **Data Visualization & Reporting** | [![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?logo=matplotlib&logoColor=black)](https://matplotlib.org/) [![Pillow](https://img.shields.io/badge/Pillow-945025)](https://python-pillow.org/) [![JSON](https://img.shields.io/badge/JSON-000000?logo=json&logoColor=white)](https://www.json.org/) [![ReportLab](https://img.shields.io/badge/ReportLab-002F6C)](https://www.reportlab.com/) |
| **Infrastructure & Version Control**| [![Google Colab](https://img.shields.io/badge/Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/) [![Google Drive](https://img.shields.io/badge/Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/) [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/) |

### Prerequisites

* **Python** 3.8+
* **CUDA** compatible GPU (Highly recommended for inference/training)
* **Git**

**CLIP**
```bash
!pip install -q torch torchvision transformers pillow matplotlib tqdm
```

```python
MODEL_NAME = "openai/clip-vit-base-patch16"
```

**Grounding DINO**
```bash
!pip -q install "transformers>=4.51,<4.58" "accelerate>=0.30,<1.0" "pillow>=10,<12" matplotlib
```

```python
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
```

**Florence-2**
```bash
!pip install -q transformers==4.49.0 accelerate==0.30.1 einops==0.8.0 datasets pillow matplotlib requests
```

```python
model_id = "microsoft/Florence-2-large"
```

## 🧮 Mathematical Framework

**Standard Performance Metrics:** Prompt sensitivity, prompt consistency, per-category error, and per-prompt error are derived from these base metrics and reported in the aggregate JSON for each model:

**Pixel Error (L2 Norm):** The fundamental accuracy of the spatial grounding task is quantified using the **Euclidean displacement** between the model's **prediction (pred)** and the annotated **ground truth (gt)**. The absolute pixel error is described by:

$$E_{pixel} = \sqrt{(x_{pred} - x_{gt})^2 + (y_{pred} - y_{gt})^2}\tag{1}$$

where $(x_{pred}, y_{pred})$ are the coordinates of the predicted grasp point, and $(x_{gt}, y_{gt})$ are the coordinates of the ground-truth annotation.

**Normalized Pixel Error:** To ensure equitable comparison across varying image resolutions, the **pixel error is normalized** against the **physical dimensions (width $\times$ height) of the image**. The normalized pixel error is described by:

$$E_{norm} = \frac{E_{pixel}}{\sqrt{w^2 + h^2}}\tag{2}$$

where $w$ and $h$ are the width and height of the image, respectively, defining the diagonal length of the spatial domain.

**Success Rate:** The **functional success** of the model's localization is evaluated as a **binary thresholding problem**. The aggregate success rate is described by:

$$S_{rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[E_{pixel, i} < \tau]\tag{3}$$

where $N$ is the total number of evaluated samples, $\tau$ is the acceptable pixel error threshold (e.g., `SUCCESS_THRESHOLD = 50`), and $\mathbf{1}[\cdot]$ is the indicator function which equals $1$ if the condition is met and $0$ otherwise.

## Project Architecture
![Project Architecture](outputs/florence2_inference_outputs/introductory_figure.png)

## 📊 Preliminary Results
![Comparative evaluation of all three models](outputs/evaluation/fig_combined_4panel.png)

CLIP:
![Data visualization of Clip](outputs/Clip_v3_patch16/clip_patch_final.png)
FLORENCE-2:
![Data visualization of FLORENCE-2](outputs/florence2_preliminary_results/florence2_data_visualization.png)

## 🎓 Academic Context

- **Course**: ENGR 521 A SP 26 Machine Learning for Engineering Project
- **Institution**: University of Washington
- **Program**: Graduate Certificate in Artificial Intelligence and Machine Learning for Engineering
- **Quarter**: Spring 2026

## 🤝 Acknowledgements
- **Dr. Michelle Hickner** - Assistant Teaching Professor, Department of Mechanical Engineering
- **Morgan Sanger** - Teaching Assistant
- **Zepu Wang** - Teaching Assistant
- [CLIP](https://github.com/openai/CLIP) — OpenAI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — IDEA Research
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) — Microsoft
- AI tools used for debugging

## 📚 Bibliography

[1] M. Hickner, ENGR 521 A SP 26 Machine Learning for Engineering Project. UW Canvas: Course Materials, 2026.

[2] M. Hickner, ENGR 510 A AU 25 Foundations Of Machine Learning For Engineering. UW Canvas: Course Materials, 2025.

[3] S. Fresca, ENGR 515 A Wi 26 Data-Driven Optimization. UW Canvas: Course Materials, 2026.

[4] S. Fresca, ENGR 520 A Sp 26 Physics-Informed Machine Learning. UW Canvas: Course Materials, 2026.

[5] S. L. Brunton and J. N. Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge: Cambridge University Press, 2021.

[6] S. L. Brunton. Optimization: A Bootcamp for Machine Learning, Inverse Problems, and Control. Course Manuscript, University of Washington, 2025.

[7] S. Brunton, "Physics Informed Machine Learning: High Level Overview of AI and ML in Science and Engineering," 2024. [Online]. Available: https://www.youtube.com/watch?v=JoFW2uSd3Uo.

[8] AllenAI, "pixmo-points," Hugging Face, 2024. [Online]. Available: https://huggingface.co/datasets/allenai/pixmo-points.

[9] Ultralytics, "How to Run Microsoft Florence-2 with Ultralytics for Visual Reasoning, OCR & Object Detection Tasks," 2025. [Online]. Available: https://www.youtube.com/watch?v=ojoYESWLw5Q | [Colab Notebook](https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-use-florence-2-for-object-detection-image-captioning-ocr-and-segmentation.ipynb)

[10] A. Deshpande et al., "GraspMoLMo: Language-Guided Grasp Point Selection with Vision-Language Models," arXiv preprint, 2025. [Online]. Available:https://arxiv.org/abs/2505.13441
