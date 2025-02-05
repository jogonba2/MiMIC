<h1 align="center">MiMIC 🎭</h1>

<p align="center">
    <a href="LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-CC_BY_NC_ND_4.0-red">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0-orange">
    </a>
    <!-- <img alt="Vision Language Models" src="https://img.shields.io/badge/TBD-yellow"> -->
    <!-- <img alt="Image Generation Models" src="https://img.shields.io/badge/TBD-yellow"> -->
    <!-- <img alt="Domains" src="https://img.shields.io/badge/Domains-Wikipedia-yellow"> -->
    <img alt="Languages" src="https://img.shields.io/badge/Languages-en%2Ces-green">
    <img alt="Subtasks" src="https://img.shields.io/badge/Tasks-detection-blue">
    <a href="https://www.symanto.com/">
        <img alt="Organizers" src="https://img.shields.io/badge/Organizers-Symanto-violet">
    </a>
</p>

<p align="center">
    <a href="https://groups.google.com/u/3/g/mimic-shared-task"><img src="https://fonts.gstatic.com/s/i/productlogos/groups/v9/web-48dp/logo_groups_color_1x_web_48dp.png" alt="Google Groups" style="width: 6%; height: auto;"></a>
    <a href="mailto:organizers.mimic@gmail.com"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Gmail_icon_%282020%29.svg/512px-Gmail_icon_%282020%29.svg.png?20221017173631" alt="Google Groups" style="width: 6%; height: auto;"></a>
</p>


<h3 align="center"><b>Multi-Modal AI Content Detection</b></h3>
</br>

The **MiMIC: Multi-Modal AI Content Detection** shared task will take place as part of **IberLEF 2025**, the **7th Workshop on Iberian Languages Evaluation Forum at the SEPLN 2025 Conference**, which will be held in **Zaragoza**, **Spain** on **September**, **2024**.

MiMIC is an extension of AuTexTification and IberAuTexTification, focused on multi-modal machine-generated content detection. This first edition focuses specifically on (text, image) pairs, as they are the most prevalent in the web. It consists of two subtasks focused on the same detection task, both for English and Spanish. 

For all the information about the shared task (description, how to download the dataset, constraints, etc.), please, refer to the [webpage](https://sites.google.com/view/mimic-2025/home).

## 📢 Anouncements

- **The repository has been created at 05/02/2025**

## 🛠️ Subtasks

MiMIC consists of two subtasks focused on the same detection task, both for English and Spanish:

**Subtask 1 - Multimodal Machine Generated Content Detection in English**: Participants must detect whether a pair of (image, **English** text) has been fully or partially created by a human or generative models. 

**Subtask 2 - Multimodal Machine Generated Content Detection in Spanish**: Participants must detect whether a pair of (image, **Spanish** text) has been fully or partially created by a human or generative models.

Both subtasks involve multiclass classification of (text, image) pairs into four categories:  fully-generated, fully-human, image-generated, and text-generated. 

With these two subtasks, our goal is to explore (i) the feasibility of detecting multimodal machine-generated content, (ii) the unique challenges posed by each modality in identifying generated content, and (iii) if text and image signals complement each other to help detectors.

Participants are not limited to multimodal approaches, i.e., uni-modal approaches are also permitted, as understanding how multimodal signals work compared to uni-modal signals is essential.

The datasets will include text and images from Wikipedia datasets used to train vision-language models such as [google/wit](https://huggingface.co/datasets/google/wit)

# 👀 What is this repo for?
This repo contains code to run the baselines, evaluate your predictions and check the format of your submissions for both subtasks. Once the competition ends, we will release the code to automatically build the datasets too.

The code is prepared with extensibility in mind, so you can use it as basis to develop your own models and get some functionalities for free as CLI endpoints or config handling.

## Run Baselines
...

## Evaluation
...

## Format checking
...

# ❓ FAQ

**Q: Are there any modeling constraints in this task?**

Yes, the constraints are the following.

1) Publicly available pretrained models from the literature can be used. However, participants are only allowed to use image and text derived from the training data. That is, data augmentation, further self-supervised pre-training, or other techniques that involve the usage of additional texts and images must be done only with texts and images derived from the training data.
2) 
3) The usage of knowledge bases, lexicons and other structured data resources is also allowed.
4) 
5) Usage of data from one subtask in the other subtask is not allowed.

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

**Google groups**: [https://groups.google.com/g/iberautextification](https://groups.google.com/g/iberautextification)

**Organizers email**: [organizers.mimic@gmail.com](mailto:organizers.mimic@gmail.com)
