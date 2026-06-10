import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

class FineTunerDPO:
    def __init__(self, model_name="google/flan-t5-large", output_dir="model_output/dpo_lora"):
        """
        Stage 9: Fine-Tuning Loop
        Trains the generator using user feedback (the 'chosen' good responses) via LoRA.
        Uses Seq2SeqTrainer which is fully compatible with T5 encoder-decoder models.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        print(f"Loading Tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading Base Model for Seq2Seq Generation...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Setup LoRA
        print("Configuring LoRA parameters...")
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q", "v"],
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self, data_path="data/preference_dataset.json"):
        """
        Loads saved feedback and converts it into prompt->chosen training pairs.
        We train the model to generate the 'chosen' (good) response for each prompt.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No feedback data found at {data_path}")
            
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        if len(data) == 0:
            raise ValueError("Feedback dataset is empty!")
        
        # For each feedback entry, we teach the model:
        # Input = prompt, Output = chosen (the good response)
        prompts = [item['prompt'] for item in data]
        targets = [item['chosen'] for item in data]  # Train on good responses only
        
        hf_dataset = Dataset.from_dict({
            "prompt": prompts,
            "target": targets
        })
        
        print(f"Loaded {len(hf_dataset)} training examples for fine-tuning.")
        return hf_dataset

    def tokenize_data(self, dataset):
        """Tokenizes prompts as inputs and chosen responses as labels."""
        def preprocess(examples):
            inputs = self.tokenizer(
                examples["prompt"], 
                max_length=256, 
                truncation=True, 
                padding="max_length"
            )
            targets = self.tokenizer(
                examples["target"], 
                max_length=150, 
                truncation=True, 
                padding="max_length"
            )
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        return dataset.map(preprocess, batched=True, remove_columns=["prompt", "target"])

    def train(self, data_path="data/preference_dataset.json"):
        print("--- Starting Fine-Tuning Loop ---")
        dataset = self.load_dataset(data_path)
        tokenized_dataset = self.tokenize_data(dataset)
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            logging_steps=5,
            num_train_epochs=3,
            save_strategy="epoch",
            use_cpu=not torch.cuda.is_available(),
            predict_with_generate=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        print("Executing LoRA Fine-tuning on your chosen responses...")
        trainer.train()
        
        print("Saving LoRA Adapters...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"--- Training Complete. Adapters saved to {self.output_dir} ---")

if __name__ == "__main__":
    tuner = FineTunerDPO()
    tuner.train()
