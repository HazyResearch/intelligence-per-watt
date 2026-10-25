from __future__ import annotations

from pathlib import Path

from .config import TrainConfig
from .dataprep import IGNORE_INDEX, load_examples, split_examples


def build_peft_model(config: TrainConfig):
    """Load the base CausalLM + tokenizer and wrap the model with a LoRA adapter (peft)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    if config.gradient_checkpointing:
        # With a frozen base + LoRA, checkpointing needs the input embeddings to produce grads.
        peft_model.enable_input_require_grads()
    return peft_model, tokenizer


def run(config: TrainConfig) -> Path:
    """Tokenize → split → LoRA-train via SFTTrainer → save the adapter; returns the adapter dir."""
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = build_peft_model(config)
    examples = load_examples(config.data_path, tokenizer, max_seq_len=config.max_seq_len)
    train_ex, val_ex = split_examples(examples, val_frac=config.val_frac, seed=config.seed)
    train_ds = Dataset.from_list(train_ex)
    eval_ds = Dataset.from_list(val_ex) if val_ex else None

    out_dir = Path(config.out_dir)
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for LoRA + checkpointing
        seed=config.seed,
        max_length=config.max_seq_len,
        logging_steps=1,
        report_to=[],
        # We pre-tokenized (input_ids + labels already present): tell TRL not to re-prepare the dataset.
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # Pass the tokenizer explicitly (TRL>=1.0 renamed `tokenizer` -> `processing_class`). Without it
        # SFTTrainer calls AutoProcessor.from_pretrained, which for multimodal-capable families (Qwen3.5,
        # Gemma4) pulls a video/image processor needing torchvision and errors on a text-only install.
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir
