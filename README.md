Fine-Tuning a Medical Diagnosis LLM on Apple Silicon using MLX

This project demonstrates how to fine-tune a pre-trained large language model (LLM) on a domain-specific dataset — in this case, a collection of medical symptoms and their corresponding diagnoses. The fine-tuning was performed using Apple’s MLX framework on an Apple Silicon (M-series) machine.

Why Fine-Tune?
Fine-tuning allows us to adapt a general-purpose LLM to perform well on a narrow task or specialized domain. Instead of training a model from scratch — which is resource-intensive — we build upon the language understanding of a pre-trained model. This results in faster training, better performance on niche data, and reduced hardware requirements.

Techniques Used

1. Instruction Fine-Tuning

The model was trained using prompt-response style examples that teach the LLM how to interpret symptoms and return probable diagnoses.

2. Parameter-Efficient Fine-Tuning (LoRA & QLoRA)
	•	LoRA (Low-Rank Adaptation): Injects small trainable matrices into the model to reduce training overhead.
	•	QLoRA: Combines LoRA with quantization to further reduce memory usage — enabling training on consumer-grade GPUs.

Environment Setup

python -m venv .venv
source .venv/bin/activate
pip install -U mlx-lm pandas huggingface_hub "huggingface_hub[cli]"
huggingface-cli login  # Enter your access token

Dataset Preparation
	•	Original format: CSV with label (diagnosis) and text (symptoms)
	•	Reformatted into natural language prompt-response pairs
	•	Saved in train.jsonl, test.jsonl, and valid.jsonl for MLX training

Example entry:

{
  "text": "You are a medical diagnosis expert. Symptoms: 'I have been experiencing ...'. Question: 'What is the diagnosis I have?'. Response: You may be diagnosed with Typhoid."
}

Model Selection

Used mlx-community/Ministral-8B-Instruct-2410-4bit, a quantized 8B parameter model optimized for Apple Silicon.
Download with:

huggingface-cli download mlx-community/Ministral-8B-Instruct-2410-4bit

Fine-Tuning

Fine-tune with LoRA using the following command:

python -m mlx_lm.lora \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --data data \
    --train \
    --fine-tune-type lora \
    --batch-size 4 \
    --num-layers 16 \
    --iters 1000 \
    --adapter-path adapters

Evaluation

Compare outputs before and after fine-tuning:

Without adapters (baseline)
python -m mlx_lm.generate \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --prompt "Symptoms: ... Question: What could be the diagnosis?"

With adapters (fine-tuned)
python -m mlx_lm.generate \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --adapter-path adapters \
    --prompt "Symptoms: ... Question: What could be the diagnosis?"

The fine-tuned model produced more direct, diagnosis-focused outputs.

Results
	•	Fine-tuned LLM generated concise, relevant responses aligned with training data
	•	Reduced memory usage using LoRA and QLoRA
	•	Successfully trained and deployed on MacBook Pro with M3 Pro and 18GB RAM

Repo Structure

├── dataset/
│   └── symptoms_diagnosis.csv
├── data/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── valid.jsonl
├── adapters/
│   └── ...  # Fine-tuned LoRA weights
├── model/
│   └── fine-tuned_Ministral-8B/  # Fused model

Conclusion

This project showcases how Apple Silicon, paired with the MLX framework and LoRA/QLoRA techniques, can be a powerful and efficient platform for domain-specific LLM fine-tuning.
