# XLSR-Mamba Setup and Inference

This document guides you through setting up the environment, installing dependencies, and running inference using the XLSR-Mamba architecture. 
[Doc for more model design details.](https://drive.google.com/file/d/152VSab4DwWj9vnLbfLa8cIlfC8-mZWjN/view?usp=drive_link)

## Setup Environment

### Step 1: Create and Activate Anaconda Environment

First, create and activate a new conda environment:

```bash
conda create -n XLSR_Mamba python=3.10
conda activate XLSR_Mamba
```

### Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Install Fairseq

Clone and install Fairseq from source:

```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
pip install --editable ./
cd ..
```

**Note**: If installation issues occur, consider temporarily downgrading pip. After installing Fairseq, upgrade pip again:

```bash
pip install --upgrade pip
```

## Pretrained Models

- **XLSR Model**: [Download XLSR pretrained model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)
- **DualMamba Models**: [Download pretrained DualMamba models](https://drive.google.com/file/d/14e6d6z4KTt5ZDPTjh5PJloNzQAxivyEu/view?usp=sharing)

Place the downloaded pretrained models into your working directory or specify their paths accordingly.

## Directory Structure

Testing files can be downloaded from here: [Link](https://drive.google.com/drive/folders/0AGlQrnaCh0OrUk9PVA)

Create the following directories and file structure before running inference:

```plaintext
|audio
|-real
|--real_0.wav
|-fake
|--fake_0.wav

|model
|-model_0.pt
```

## Run Eval

Execute the following command to run inference:

```bash
python deepfake_eval.py
```

Ensure your environment is activated (`conda activate XLSR_Mamba`) and dependencies are properly installed before running this command.

## Test Log

![image](https://github.com/user-attachments/assets/691950f3-5a1c-433f-abca-caf6d2997160)


