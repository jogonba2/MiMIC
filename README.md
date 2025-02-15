<h1 align="center">MiMIC 🎭</h1>

<p align="center">
    <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-red">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0-orange">
    </a>
    <img alt="Vision Language Models" src="https://img.shields.io/badge/Vision_Language_Models-TBD-yellow">
    <img alt="Image Generation Models" src="https://img.shields.io/badge/Image_Generation_Models-TBD-yellow">
    <img alt="Domains" src="https://img.shields.io/badge/Domains-TBD-yellow">
    <img alt="Languages" src="https://img.shields.io/badge/Languages-en%2Ces-green">
    <img alt="Subtasks" src="https://img.shields.io/badge/Tasks-detection-blue">
    <a href="https://www.symanto.com/">
        <img alt="Organizer" src="https://img.shields.io/badge/Organizer-Symanto-violet">
    </a>
</p>

<p align="center">
    <a href="https://groups.google.com/u/3/g/mimic-shared-task"><img src="https://fonts.gstatic.com/s/i/productlogos/groups/v9/web-48dp/logo_groups_color_1x_web_48dp.png" alt="Google Groups" style="width: 6%; height: auto;"></a>
    <a href="mailto:organizers.mimic@gmail.com"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Gmail_icon_%282020%29.svg/512px-Gmail_icon_%282020%29.svg.png?20221017173631" alt="Google Groups" style="width: 6%; height: auto;"></a>
</p>


<h3 align="center"><b>Multi-Modal AI Content Detection</b></h3>
</br>

The **MiMIC: Multi-Modal AI Content Detection** shared task will take place as part of **IberLEF 2025**, the **7th Workshop on Iberian Languages Evaluation Forum at the SEPLN 2025 Conference**, which will be held in **Zaragoza**, **Spain** on **September**, **2025**.

MiMIC is an extension of AuTexTification and IberAuTexTification, focused on multi-modal machine-generated content detection. This first edition focuses specifically on (text, image) pairs, as they are the most prevalent in the web. It consists of two subtasks focused on the same detection task, both for English and Spanish. 

For all the information about the shared task (description, how to download the dataset, constraints, etc.), please, refer to the [webpage](https://sites.google.com/view/mimic-2025/home).

## 📢 Anouncements

- **The repository has been created at 05/02/2025**
- **Endpoints for running experiments, evaluate, and format checking has been released at 15/02/2025**

## 🛠️ Subtasks

MiMIC consists of two subtasks focused on the same detection task, both for English and Spanish:

**Subtask 1 - Multimodal Machine Generated Content Detection in English**: Participants must detect whether a pair of (image, **English** text) has been fully or partially created by a human or generative models. 

**Subtask 2 - Multimodal Machine Generated Content Detection in Spanish**: Participants must detect whether a pair of (image, **Spanish** text) has been fully or partially created by a human or generative models.

Both subtasks involve multiclass classification of (text, image) pairs into four categories:  fully-generated, fully-human, image-generated, and text-generated. 

With these two subtasks, our goal is to explore (i) the feasibility of detecting multimodal machine-generated content, (ii) the unique challenges posed by each modality in identifying generated content, and (iii) if text and image signals complement each other to help detectors.

Participants are not limited to multimodal approaches, i.e., uni-modal approaches are also permitted, as understanding how multimodal signals work compared to uni-modal signals is essential.

# 👀 What is this repo for?
This repo contains code to run the baselines, evaluate your predictions and check the format of your submissions for both subtasks. Once the competition ends, we will release the code to automatically build the datasets too.

The code is prepared with extensibility in mind, so you can use it as basis to develop your own models and get some functionalities for free as CLI endpoints or config handling.

## Get started
Install the requirements: 
```bash
pip install -r requirements.txt
```

## Run Baselines

You can run the baselines with the `run-experiment` endpoint:
```bash
python -m mimic.cli run-experiment \
--subtask 1 \
--config-path ./etc/subtask_1_baselines.yaml \
--dataset-path ./mimic-dataset \
--team-name baselines \
--output-path ./baseline-predictions
```

You can define new models in `mimic/models` and use a similar YAML config structure to the one employed for the baselines in `etc/`. Then, with `run-experiment` you can train and evaluate your models.

This endpoint also saves the dataset specified in `--dataset-path` under a subfolder named `ground_truth` within the output path specified in `--output-path`. So, you can use this folder to evaluate the outputs of your model by using the `evaluate` endpoint.

## Evaluation

You can evaluate your model's predictions with the `evaluate` endpoint:

```bash
python -m mimic.cli evaluate \
--subtask 1 \
--submissions-dir ./evaluation_sample/submissions \
--ground-truth-dir ./evaluation_sample/ground_truth \
--output-dir ./ranking_results
```

## Format checking

You can check the format of your final submission with the `check-format` endpoint:
```bash
python -m mimic.cli check-format \
--submission-file ./evaluation_sample/submissions/baselines/subtask_1/vlm--pixtral-12-icl.jsonl \
--ground-truth-file evaluation_sample/ground_truth/subtask_1/truth.jsonl
```
Make sure your submission files are valid before submitting them to the competition!

## Explore a dataset

You can explore a dataset with the streamlit UI by running this command:
```bash
python -m streamlit run mimic/explore/app.py
```

## Build a dataset

> [!IMPORTANT]  
> This feature will be released after the competition ends, keep posted! 🤓

It will be as easy as calling the `generate_dataset` endpoint:

```python
python -m mimic.cli generate-dataset \
--n-samples 200 \
--run-name my-dataset \
--language es \
--text-models ... \
--image-models ...
```

# ❓ FAQ

**Q: Are there any modeling constraints in this task?**

Yes, the constraints are the following.

1) Publicly available pretrained models from the literature can be used. However, participants are only allowed to use image and text derived from the training data. That is, data augmentation, further self-supervised pre-training, or other techniques that involve the usage of additional texts and images must be done only with texts and images derived from the training data.
2) The usage of knowledge bases, lexicons and other structured data resources is also allowed.
3) Usage of data from one subtask in the other subtask is not allowed.

**Q: How many submissions can we submit?**

3 submissions are allowed per team for each subtask.

**Q: Must we participate in all subtasks or just one of them?**

Participants are free to participate in any of the two subtasks.

# 🚀 Organizers

- José Ángel González (jose.gonzalez@symanto.com) - Symanto Research, Valencia, Spain
- Areg Sarvazyan (areg.sarvazyan@symanto.com) - Symanto Research, Valencia, Spain
- Angelo Basile (angelo.basile@symanto.com) - Symanto Research, Valencia, Spain
- Ian Borrego (ian.borrego@symanto.com) - Symanto Research, Valencia, Spain
- Mara Chinea (mara.chinea@symanto.com) - Symanto Research, Valencia, Spain
- Francisco Rangel (francisco.rangel@symanto.com) - Symanto, Valencia, Spain

# #️⃣ Social

**Google groups**: [https://groups.google.com/u/3/g/mimic-shared-task](https://groups.google.com/u/3/g/mimic-shared-task)

**Organizers email**: [organizers.mimic@gmail.com](mailto:organizers.mimic@gmail.com)