from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import huggingface_hub
import torch

from dataset_cleaning import token_filter, limit_post_id_occurrences

def main() -> None:
    """
    ======================================
    SET UP ENVIRONMENT
    ======================================
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    huggingface_hub.login(token = "<token>")

    """
    ======================================
    HYPERPARAMS
    ======================================
    """
    MODEL_ID = "AuriLab/gpt-bi"
    PUSH_REPO = "AuriLab/gpt-bi-instruct"
    MAX_OCCURRENCES = 5
    EPOCHS = 1
    BATCH_SIZE = 2
    LEARNING_RATE = 3e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    LOGGING_SAVE_EVAL_STEPS = 500
    SAVE_TOTAL_LIMIT = 3

    """
    ======================================
    MODEL & TOKENIZER
    ======================================
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

    """
    ======================================
    DATASET PREPROCESSING
    ======================================
    """
    dataset = load_dataset("AuriLab/combined_oasst_eu")

    dataset = dataset.filter(token_filter, fn_kwargs={"block_size" : tokenizer.model_max_length, "tokenizer" : tokenizer})
    dataset = limit_post_id_occurrences(dataset, max_occurrences = MAX_OCCURRENCES)

    """
    ======================================
    PREPARATE TRAINING
    ======================================
    """
    training_args = SFTConfig(
        output_dir = "./gpt-bi-instruct",
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        logging_steps = LOGGING_SAVE_EVAL_STEPS,
        save_steps = LOGGING_SAVE_EVAL_STEPS,
        eval_strategy = "steps",
        eval_steps = LOGGING_SAVE_EVAL_STEPS,
        
        weight_decay = WEIGHT_DECAY,
        warmup_ratio = WARMUP_RATIO,
        lr_scheduler_type = "cosine",
        save_total_limit = SAVE_TOTAL_LIMIT,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        fp16 = True,
    )

    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        processing_class = tokenizer,
    )

    trainer.train()

    """
    ======================================
    PUSH TO HUB
    ======================================
    """
    model.push_to_hub(PUSH_REPO)
    tokenizer.push_to_hub(PUSH_REPO)

if __name__ == "__main__":
    main()