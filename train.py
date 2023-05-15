from IPython.core.application import default
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from itertools import repeat
import argparse

def read_data(path, tokenizer):
    input_lines = []
    label_lines = []
    with open(path, "r") as file:
        for line in file:
            example = line.strip().split('\t')
            if len(example) == 2:
              input_lines.append((example[0]))
              label_lines.append((example[1]))

    dict_obj = {'inputs': input_lines, 'labels': label_lines}
    dataset = Dataset.from_dict(dict_obj)

    tokenized_datasets = dataset.map(preprocess_function, fn_kwargs={"tokenizer":tokenizer}, batched=True, remove_columns=['inputs'], num_proc=8)

    return tokenized_datasets

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=1024, truncation=True, padding=True)
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs

def train(arg):
    tokenizer = AutoTokenizer.from_pretrained(arg.pretrain)  
    model = AutoModelForSeq2SeqLM.from_pretrained(arg.pretrain)
    if arg.gpus:
        model.to('cuda')
    
    train_dataset = read_data(arg.train, tokenizer)
    val_dataset = read_data(arg.val, tokenizer)


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    training_args = Seq2SeqTrainingArguments("tmp/",
                                          do_train=True,
                                          do_eval=True,
                                          num_train_epochs=arg.epoch,
                                          learning_rate=arg.lr,
                                          warmup_ratio=0.05,
                                          weight_decay=0.01,
                                          per_device_train_batch_size=arg.batch,
                                          per_device_eval_batch_size=arg.batch,
                                          logging_dir='./log',
                                          group_by_length=True,
                                          save_strategy="epoch",
                                          save_total_limit=3,
                                          eval_steps=500,
                                          evaluation_strategy="steps",
                                          fp16=True,
                                          )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset= val_dataset,
        data_collator=data_collator,
    )

    trainer.train()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default = "VietAI/vit5-base", type=str, help="Pretrain Path")
    parser.add_argument("--gpus", default= True, action="store_true", help="Use GPUS")
    parser.add_argument("--lr", default=1e-5, help="Learning Rate")
    parser.add_argument("--batch",default=4 , help="Num Batch")
    parser.add_argument("--epoch", default=5, help="Num Epoch")
    parser.add_argument("--train",default= "data/train.tsv" , help="Data train")
    parser.add_argument("--val",default= "data/val.tsv" , help="Data val")

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()
    train(args)




