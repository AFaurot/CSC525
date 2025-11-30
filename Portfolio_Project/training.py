import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)

# Device Auto-Detection -- much faster on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected using: {device}")

# Load Datasets
print("Loading SQuAD v2 dataset...")
squad = load_dataset("squad_v2")

# I created a custom addendum dataset in SQuAD like format with fishing-related Q&A.
# There are only a few examples, but in addition to SQuAD v2, it should help the model learn fishing-related QA.
print("Loading CPW addendum dataset from cpw_addendum.json...")
cpw_nested = load_dataset(
    "json",
    data_files="cpw_addendum.json",  # make sure this file is in the working directory
    field="data",                    # use the `data` field of your JSON
)["train"]


def flatten_cpw(cpw_dataset):
    """Flatten nested CPW JSON (SQuAD-style) into a flat table like SQuAD v2."""
    titles = []
    contexts = []
    questions = []
    answers_list = []
    ids = []
    is_impossible_list = []

    for example in cpw_dataset:
        title = example["title"]
        for para in example["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                titles.append(title)
                contexts.append(context)
                questions.append(qa["question"])
                ids.append(qa["id"])

                answers_list.append({
                    "text": [a["text"] for a in qa["answers"]],
                    "answer_start": [a["answer_start"] for a in qa["answers"]],
                })

                is_impossible_list.append(qa.get("is_impossible", False))

    return Dataset.from_dict({
        "id": ids,
        "title": titles,
        "context": contexts,
        "question": questions,
        "answers": answers_list,
        "is_impossible": is_impossible_list,
    })


print("Flattening CPW addendum to SQuAD-like format...")
cpw_flat = flatten_cpw(cpw_nested)

# Remove column not present in SQuAD v2 -- This column is not needed for training
if "is_impossible" in cpw_flat.column_names:
    cpw_flat = cpw_flat.remove_columns(["is_impossible"])

# Now cast safely (fixes int64 incompatibility issue by casting to int32)
cpw_flat = cpw_flat.cast(squad["train"].features)


print("Combining SQuAD v2 train split with CPW addendum...")
combined_train_raw = concatenate_datasets([squad["train"], cpw_flat])
print(f"SQuAD train size: {len(squad['train'])}")
print(f"Combined train size: {len(combined_train_raw)}")

# Load Tokenizer and Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# Preprocessing Function
def prepare_squad(examples):
    # Strip whitespace from questions
    questions = [q.strip() for q in examples["question"]]

    # Tokenize question + context pairs
    model_inputs = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
    )
    offset_mapping = model_inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]

        # Case: No answer exists - Relevant for SQuAD v2
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Determine which tokens correspond to context
        sequence_ids = model_inputs.sequence_ids(i)
        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # Identify token start/end positions
        start_pos = None
        end_pos = None

        for idx in range(ctx_start, ctx_end + 1):
            if offsets[idx][0] <= start_char < offsets[idx][1]:
                start_pos = idx
            if offsets[idx][0] < end_char <= offsets[idx][1]:
                end_pos = idx

        # Fallback to CLS if we couldn't find a match
        if start_pos is None or end_pos is None:
            start_pos = 0
            end_pos = 0

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    model_inputs["start_positions"] = start_positions
    model_inputs["end_positions"] = end_positions

    return model_inputs


# Tokenization / Alignment
print("Tokenizing and aligning labels for combined train set...")
train_dataset = combined_train_raw.map(
    prepare_squad,
    batched=True,
    remove_columns=combined_train_raw.column_names,
)

print("Tokenizing and aligning labels for SQuAD validation set...")
eval_dataset = squad["validation"].map(
    prepare_squad,
    batched=True,
    remove_columns=squad["validation"].column_names,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Training Arguments and Hyperparameters
training_args = TrainingArguments(
    output_dir="./model_output",
    eval_strategy="steps",
    eval_steps=10000,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=500,
    fp16=torch.cuda.is_available(),  # mixed precision on GPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
print("Beginning training...")
trainer.train()
print("Training completed!")

print("Saving model and tokenizer to ./model_output...")
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")
print("Model saved to ./model_output")

# Quick Check with QA Pipeline
print("\nPreparing QA pipeline for testing...")
qa_pipeline = pipeline(
    "question-answering",
    model="./model_output",
    tokenizer="./model_output",
    device=0 if device == "cuda" else -1,
)

# Example fishing-related context
context = """
Colorado fishing regulations require anglers to possess a valid fishing license.
A second rod stamp is needed only if you plan to use more than one rod at a time.
The trout bag limit at most Colorado lakes is 4 fish per day unless otherwise posted.
"""

questions = [
    "Do I need a second rod stamp?",
    "What is the trout bag limit?",
    "Do I need a fishing license?",
]

print("\n Sample Question-Answering Results:")
for q in questions:
    result = qa_pipeline(question=q, context=context)
    print(f"\nQ: {q}\nA: {result['answer']} (score: {result['score']:.3f})")
