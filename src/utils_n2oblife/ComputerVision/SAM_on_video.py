# Notebook: fine-tune SAM (segment anything) on a custom dataset

# In this notebook, we'll reproduce the [MedSAM](https://github.com/bowang-lab/MedSAM) project, which fine-tunes [SAM](https://huggingface.co/docs/transformers/main/en/model_doc/sam) on a dataset of medical images. For demo purposes, we'll use a toy dataset, but this can easily be scaled up.

# Resources used to create this notebook (thanks 🙏):
# * [Encode blog post](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)
# * [MedSAM<< repository](https://github.com/bowang-lab/MedSAM).

## Set-up environment

# # We first install 🤗 Transformers and 🤗 Datasets.

