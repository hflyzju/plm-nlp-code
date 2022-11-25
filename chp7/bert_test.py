

from transformers import BertModel, BertConfig, BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
# tokenizer = BertTokenizerFast.from_pretrained('/Users/huxiang/Documents/pretrain_models/bert-tiny/')


config = BertConfig()
bert = BertModel(config)

print(bert.config())