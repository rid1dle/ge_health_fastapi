import os
import json
import shutil
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd

# import mlflow
import gc

from .train_base import BaseTrainer

# EXPORT_FOLDER = str(Path(__file__).parent.parent) + "/export"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class LLAMAModel(BaseTrainer):
    def __init__(
        self,
        search_space: dict,
        user_cfg: dict,
        dataset: str,
        model_id: str,
        trial_id: str,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.user_config = user_cfg
        self.lr = search_space["learning_rate"]
        self.warm_step = search_space["warm_step"]
        self.batch_size = user_cfg.get("batch_size", 1)
        self.num_epochs = user_cfg.get("epochs", 1)
        self.model_or_path = user_cfg.get("model_or_path", "NousResearch/Llama-2-7b-hf")
        print(self.model_or_path)

        dataset = self.preprocess_dataset(dataset)
        self.dataset = dataset
        self.num_steps = int(
            (
                (len(dataset) // self.batch_size)
                + int(bool(len(dataset) % self.batch_size))
            )
            * self.num_epochs
        )
        self.output_dir = os.path.join(os.getcwd(), model_id, f"trial_{trial_id}", "1")
        os.makedirs(self.output_dir, exist_ok=True)

        # config_json = {
        #     "model_type": "llm",
        #     "input_features": [
        #         {"name": "input_text", "type": "text"},
        #         {"name": "output_len", "type": "number"},
        #         {"name": "temperature", "type": "number"},
        #         {"name": "topk", "type": "number"},
        #         {"name": "topp", "type": "number"},
        #     ],
        #     "output_features": [{"name": "generated_text", "type": "text"}],
        # }
        # json.dump(
        #     config_json, open(os.path.join(self.output_dir, "../config.json"), "w")
        # )

        print(f"CUDA available: {torch.cuda.is_available()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        bnb_4bit_compute_dtype = "float16"
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Enable fp16/bf16 training (set bf16 to True with an A100)
        self.fp16 = False
        self.bf16 = False

        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16: accelerating training with bf16=True"
                )
                print("=" * 80)
                self.bf16 = True

        self.optimizer = self.user_config.get("optimizer", "paged_adamw_32bit")

        self.model = self._compile_model()

        # shutil.copy(
        #     f"{EXPORT_FOLDER}/llm/model.py", os.path.join(self.output_dir, "model.py")
        # )
        # self.export_model(model_id)

    def preprocess_dataset(self, dataset: str):
        df = pd.read_csv(dataset)
        df.dropna(inplace=True)
        input_col = self.user_config["targets"][0]
        df = df[[input_col]]

        df = Dataset.from_pandas(df)
        return df

    def _compile_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_or_path,
            config=self.peft_config,
            quantization_config=self.bnb_config,
            device_map="auto",
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_or_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.data = self.dataset.map(
            lambda samples: self.tokenizer(samples["text"]), batched=True
        )
        return model

    def train(self):
        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=self.batch_size,
            optim=self.optimizer,
            logging_steps=1,
            learning_rate=self.lr,
            weight_decay=0.001,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=0.3,
            warmup_steps=self.warm_step,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
        )

        print("****** STARTED TRAINING ******")
        trainer.train()
        print("****** FINISHED TRAINING ******")
        return self.post_training(trainer)

    def post_training(self, trainer):
        print("****** STARTING GARBAGE COLLECTION ******")
        gc.collect()

        print("****** DONE GARBAGE COLLECTION ******")
        if torch.cuda.is_available():
            print("****** STARTING GPU GARBAGE COLLECTION ******")
            print(
                f"""
                    BEFORE GARBAGE COLLECTION
                    TOTAL MEMORY:\t{torch.cuda.get_device_properties(0).total_memory}
                    RESERVED MEMORY:\t{torch.cuda.memory_reserved(0)}
                    ALLOCATED MEMORY:\t{torch.cuda.memory_allocated(0)}
                """
            )

            torch.cuda.empty_cache()

            print(
                f"""
                    AFTER GARBAGE COLLECTION
                    TOTAL MEMORY:\t{torch.cuda.get_device_properties(0).total_memory}
                    RESERVED MEMORY:\t{torch.cuda.memory_reserved(0)}
                    ALLOCATED MEMORY:\t{torch.cuda.memory_allocated(0)}
                """
            )
            print("****** DONE GPU GARBAGE COLLECTION ******")

        print("****** STARTING SAVING BEST MODEL ******")

        # trainer.model.save_pretrained(self.output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_or_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, self.output_dir)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(self.model_or_path)

        model.save_pretrained(f"{self.output_dir}")
        tokenizer.save_pretrained(f"{self.output_dir}")
        print(f"Model saved to {self.output_dir}")
        print("****** DONE SAVING BEST MODEL ******")

        # TODO Setup MLFlow
        # training_logs = trainer.state.log_history
        # train_metrics = {}
        # for log in training_logs:
        #     for key in log.keys():
        #         if "train" in key:
        #             train_metrics[key] = log[key]

        # mlflow.log_metrics(train_metrics)
        # train_metrics["score"] = train_metrics["train_loss"]

        # print("****** LOADED METRICS ******")
        # return train_metrics
