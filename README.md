# LLM Fine-Tuning Project

## Overview

This project demonstrates the process of fine-tuning a GPT-2 language model using both full parameter training and LoRA (Low-Rank Adaptation) techniques. The implementation includes dataset loading, model training, evaluation, and performance comparison using another LLM (DeepSeek-R1-Distill-Llama-70b) as an evaluator.

## Features

- **Dataset Handling**: Loads and processes question-answer pairs from HuggingFace datasets
- **Model Training**:
  - Full parameter fine-tuning of GPT-2
  - LoRA-based fine-tuning for efficient adaptation
- **Evaluation**:
  - Uses another LLM (DeepSeek) to evaluate model responses
  - Calculates accuracy metrics
  - Saves evaluation results for analysis

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- Additional dependencies:
  ```
  jsonlines python-dotenv lamini datasets langchain_groq numpy==1.26.4
  ```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set up your environment variables (including GROQ_API_KEY)
2. Run the script:
   ```bash
   python copy_of_llm_finetuning_official(1).py
   ```

The script will:
- Load and preprocess the dataset
- Fine-tune GPT-2 using both full parameter and LoRA methods
- Evaluate model performance
- Save the trained models and evaluation results

## Configuration

Key parameters you can adjust:
- Learning rate (`learning_rate`)
- Number of training epochs (`num_train_epochs`)
- Batch sizes (`per_device_train_batch_size`)
- LoRA configuration (rank, alpha, dropout)
- Evaluation model (`MODEL_NAME`)

## Results

The script outputs:
- Training metrics (loss, accuracy)
- Final evaluation accuracy
- CSV file with evaluation results (`eval_df.csv`)
- Saved model checkpoints

## Notes

- The script is configured to run on GPU if available
- Memory requirements are substantial for full model fine-tuning
- LoRA provides a more memory-efficient alternative
- Evaluation requires a GROQ API key for the DeepSeek model

