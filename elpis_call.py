import transformers

import examples.research_projects.wav2vec2.run_elpis as elpis

elpis.train(
    "transformers/examples/research_projects/wav2vec2/run_elpis.py",
    elpis_data_dir="../na-elpis/e919e6c711e2c6abee5a17f82bebe850",
    train_size="0.8",
    split_seed="42",
    model_name_or_path="facebook/wav2vec2-large-xlsr-53",
    dataset_config_name="tr",
    output_dir="./wav2vec2-large-xlsr-na-4",
    overwrite_output_dir=True,
    num_train_epochs="30",
    per_device_train_batch_size="4",
    per_device_eval_batch_size="4",
    gradient_accumulation_steps="2",
    learning_rate="3e-4",
    warmup_steps="500",
    evaluation_strategy="steps",
    save_steps="400",
    eval_steps="400",
    logging_steps="400",
    save_total_limit="3",
    freeze_feature_extractor=True,
    feat_proj_dropout="0.0",
    layerdrop="0.1",
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    do_train=True,
    do_eval=True)
