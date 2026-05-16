#Grasp Point Prediction
A Comparative Study of CLIP, Grounding DINO, and Florence-2 for Language-Guided Grasp-Relevant Point Prediction

## Team
- Priyadarshini Rajmohan
- Poojitha Alam
- Jannine G. D. MacGormain

## Project Structure
data/           ← pixmo_subset_v3 dataset
notebooks/      ← all model training and inference notebooks
outputs/        ← metrics and results

## Models Compared
- CLIP
- Grounding DINO  
- Florence-2

## Dataset
Pixmo subset v3 — images with language-guided grasp point annotations

## Preliminary Results
![Comparative evaluation of all three models](outputs/evaluation/fig_combined_4panel.png)

CLIP:
![Data visualization of Clip](outputs/Clip_v3_patch16/clip_patch_final.png)
FLORENCE-2:
![Data visualization of FLORENCE-2](outputs/florence2_preliminary_results/florence2_data_visualization.png)

## Acknowledgements
- [CLIP](https://github.com/openai/CLIP) — OpenAI
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — IDEA Research
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) — Microsoft
- AI tools used for debugging
