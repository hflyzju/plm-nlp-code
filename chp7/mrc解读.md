## 一、datasets使用


```python
from datasets import load_dataset, load_metric

dataset = load_dataset('squad')
```

    Reusing dataset squad (/Users/huxiang/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)



      0%|          | 0/2 [00:00<?, ?it/s]


### 1.1 也可以自定义数据
Fine-tuning with custom datasets：
https://huggingface.co/transformers/v3.2.0/custom_datasets.html

### 1.2 用datasets数据测试


```python
type(dataset)
```




    datasets.dataset_dict.DatasetDict




```python
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 87599
        })
        validation: Dataset({
            features: ['id', 'title', 'context', 'question', 'answers'],
            num_rows: 10570
        })
    })




```python
dataset['train']
```




    Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })




```python
dataset['train'][0]
```




    {'id': '5733be284776f41900661182',
     'title': 'University_of_Notre_Dame',
     'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
     'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
     'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}




```python
dataset['train'][0]['context'][515:]
```




    'Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'



## 二、transformers使用


```python
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
```


```python
tokenizer = BertTokenizerFast.from_pretrained('/Users/huxiang/Documents/pretrain_models/bert-tiny/')
```


```python
tokenizer
```




    PreTrainedTokenizerFast(name_or_path='/Users/huxiang/Documents/pretrain_models/bert-tiny/', vocab_size=30522, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})



## 三、实验


```python
import json

from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

dataset = load_dataset('squad')

tokenizer = BertTokenizerFast.from_pretrained('/Users/huxiang/Documents/pretrain_models/bert-tiny/')



def prepare_train_features(examples):
    """准备训练数据并转换为feature
    Args:
        examples: batch为n的数据，dict_keys(['id', 'title', 'context', 'question', 'answers']),
            example:{id:[id1, id2, ...], title:[title1, title2, ...],
                     context:[context1, context2, ...],
                     question:[question1, question2, ...],
                     answers:[answers1, answers2, ...]}

    """
    tokenized_examples = tokenizer(
        examples["question"],           # 问题文本
        examples["context"],            # 篇章文本
        truncation="only_second",       # 截断只发生在第二部分，即篇章
        max_length=50,                 # 设定最大长度为384
        stride=30,                     # 设定篇章切片步长为128
        return_overflowing_tokens=True, # 返回超出最大长度的标记，将篇章切成多片
        return_offsets_mapping=True,    # 返回偏置信息，用于对齐答案位置
        padding="max_length",           # 按最大长度进行补齐
    )

    print("examples.keys():", examples.keys())
    
    for key in examples.keys():
        print('examples, key=', key, 'len(examples[key]):', len(examples[key]))
    print("tokenized_examples.keys():", tokenized_examples.keys())
    for key in tokenized_examples.keys():
        print('tokenized_examples, key=', key, 'len(tokenized_examples[key]):', len(tokenized_examples[key]))
        
    # 如果篇章很长，则可能会被切成多个小篇章，需要通过以下函数建立feature到example的映射关系
    sample_mapping = tokenized_examples.get("overflow_to_sample_mapping")
    # 建立token到原文的字符级映射关系，用于确定答案的开始和结束位置
    # https://huggingface.co/course/chapter6/3
    # 标记每个实体在原文的起始位置
    # example: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22)]
    # 第1个token为(0, 2)代表占用的是原始字符串中sentence[0:2]的字
    offset_mapping = tokenized_examples.get("offset_mapping") #
    
    # 获取开始和结束位置
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取输入序列的input_ids以及[CLS]标记的位置（在BERT中为第0位）
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取哪些部分是问题，哪些部分是篇章
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 获取答案在文本中的字符级开始和结束位置
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # 获取在当前切片中的开始和结束位置
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # 检测答案是否超出当前切片的范围
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # 超出范围时，答案的开始和结束位置均设置为[CLS]标记的位置
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 将token_start_index和token_end_index移至答案的两端
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples



```

    Reusing dataset squad (/Users/huxiang/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)



      0%|          | 0/2 [00:00<?, ?it/s]


### 3.1 examples包括id，title（文章title），context（原文），question（内容），answers（包括text答案内容，以及answer_start起点）等成分


```python
tokenized_examples = prepare_train_features(dataset['train'][0:2])
```

    examples.keys(): dict_keys(['id', 'title', 'context', 'question', 'answers'])
    examples, key= id len(examples[key]): 2
    examples, key= title len(examples[key]): 2
    examples, key= context len(examples[key]): 2
    examples, key= question len(examples[key]): 2
    examples, key= answers len(examples[key]): 2
    tokenized_examples.keys(): dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
    tokenized_examples, key= input_ids len(tokenized_examples[key]): 86
    tokenized_examples, key= token_type_ids len(tokenized_examples[key]): 86
    tokenized_examples, key= attention_mask len(tokenized_examples[key]): 86
    tokenized_examples, key= offset_mapping len(tokenized_examples[key]): 86
    tokenized_examples, key= overflow_to_sample_mapping len(tokenized_examples[key]): 86



```python
tokenizer.decode(tokenized_examples['input_ids'][0])
```




    "[CLS] to whom did the virgin mary allegedly appear in 1858 in lourdes france? [SEP] architecturally, the school has a catholic character. atop the main building's gold dome is a golden statue of the virgin mary. immediately in front of the [SEP]"



#### 3.1.1 相当于各种key下面的值是一个list，并且每一个context可能有多个问题，但是每个问题只有一个答案


```python
dataset['train']
```




    Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })




```python
dataset['train']['answers'][0]
```




    {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}




```python
dataset['train'][0:3]
```




    {'id': ['5733be284776f41900661182',
      '5733be284776f4190066117f',
      '5733be284776f41900661180'],
     'title': ['University_of_Notre_Dame',
      'University_of_Notre_Dame',
      'University_of_Notre_Dame'],
     'context': ['Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
      'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
      'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'],
     'question': ['To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
      'What is in front of the Notre Dame Main Building?',
      'The Basilica of the Sacred heart at Notre Dame is beside to which structure?'],
     'answers': [{'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]},
      {'text': ['a copper statue of Christ'], 'answer_start': [188]},
      {'text': ['the Main Building'], 'answer_start': [279]}]}



### 3.2 tokenized_examples包含input_ids、token_type_ids、attention_mask、offset_mapping（每个token对应到原文的(start, end)位置），overflow_to_sample_mapping（每个sample对应到原始的第i个sentence）


```python
tokenized_examples.keys()
```




    dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping', 'start_positions', 'end_positions'])




```python
for key in tokenized_examples.keys():
    print('tokenized_examples, key=', key, 'len(tokenized_examples[key]):', len(tokenized_examples[key]))
```

    tokenized_examples, key= input_ids len(tokenized_examples[key]): 86
    tokenized_examples, key= token_type_ids len(tokenized_examples[key]): 86
    tokenized_examples, key= attention_mask len(tokenized_examples[key]): 86
    tokenized_examples, key= offset_mapping len(tokenized_examples[key]): 86
    tokenized_examples, key= overflow_to_sample_mapping len(tokenized_examples[key]): 86
    tokenized_examples, key= start_positions len(tokenized_examples[key]): 86
    tokenized_examples, key= end_positions len(tokenized_examples[key]): 86


#### 3.2.1 offset_mapping：相当于该sample的每个token对应到原始里面的sentence里面（start，end）的位置


```python
print(tokenized_examples['offset_mapping'][0])
```

    [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47), (48, 52), (53, 55), (56, 59), (59, 63), (64, 70), (70, 71), (0, 0), (0, 13), (13, 15), (15, 16), (17, 20), (21, 27), (28, 31), (32, 33), (34, 42), (43, 52), (52, 53), (54, 58), (59, 62), (63, 67), (68, 76), (76, 77), (77, 78), (79, 83), (84, 88), (89, 91), (92, 93), (94, 100), (101, 107), (108, 110), (111, 114), (115, 121), (122, 126), (126, 127), (128, 139), (140, 142), (143, 148), (149, 151), (152, 155), (0, 0)]


#### 3.2.2 start_positions和end_positions为每个sample里面答案的start和end位置，如果没有question的答案，那么为0


```python
print(tokenized_examples['start_positions'])
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 34, 28, 22, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



```python
print(tokenized_examples['end_positions'])
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 44, 38, 32, 26, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


#### 3.2.3 根据start_positions和end_positions找到原始的answer的位置


```python
i = 0
for s, e in zip(tokenized_examples['start_positions'], tokenized_examples['end_positions']):
    if s != 0 and e != 0:
        sentence_index = tokenized_examples['overflow_to_sample_mapping'][i]
        answer = dataset['train']['answers'][sentence_index]
        print('sentence_index:', sentence_index)
        print('answer:', answer)
        print('decode answer:',tokenizer.decode(tokenized_examples['input_ids'][i][s:e+1]))
    i += 1
```

    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 0
    answer: {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
    decode answer: saint bernadette soubirous
    sentence_index: 1
    answer: {'text': ['a copper statue of Christ'], 'answer_start': [188]}
    decode answer: a copper statue of christ
    sentence_index: 1
    answer: {'text': ['a copper statue of Christ'], 'answer_start': [188]}
    decode answer: a copper statue of christ
    sentence_index: 1
    answer: {'text': ['a copper statue of Christ'], 'answer_start': [188]}
    decode answer: a copper statue of christ
    sentence_index: 1
    answer: {'text': ['a copper statue of Christ'], 'answer_start': [188]}
    decode answer: a copper statue of christ
    sentence_index: 1
    answer: {'text': ['a copper statue of Christ'], 'answer_start': [188]}
    decode answer: a copper statue of christ


#### 3.2.5 sequence_ids用于找到当前切片位置：

```
# 获取在当前切片中的开始和结束位置
token_start_index = 0
while sequence_ids[token_start_index] != 1:
    token_start_index += 1
    token_end_index = len(input_ids) - 1
while sequence_ids[token_end_index] != 1:
    token_end_index -= 1
```


```python
for i in range(86):
    seq_id = tokenized_examples.sequence_ids(i)
    print('into i:', i)
    print(seq_id)
    if i > 3:
        break
```

    into i: 0
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
    into i: 1
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
    into i: 2
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
    into i: 3
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
    into i: 4
    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]


#### 3.2.6 将input_ids decode回去后，可以发现是用了sliding windows在对原文进行处理的


```python
key = 'start_positions'
for i in range(len(tokenized_examples[key])):
    if tokenized_examples['start_positions'][i] or tokenized_examples['end_positions'][i]:
        print('into:', i)
        print(tokenized_examples['start_positions'][i])
        print(tokenized_examples['end_positions'][i])
        print(tokenizer.decode(tokenized_examples['input_ids'][i]))
        print(tokenizer.decode(tokenized_examples['input_ids'][i][tokenized_examples['start_positions'][i]:tokenized_examples['end_positions'][i]]))
    if i>4:
        break
```

#### 3.2.7 offset的结果


```python
for i, offset in enumerate(tokenized_examples['offset_mapping']):
    print('i:', i)
    print('offset:', offset)
    if i > 2:
        break
```

    i: 0
    offset: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47), (48, 52), (53, 55), (56, 59), (59, 63), (64, 70), (70, 71), (0, 0), (0, 13), (13, 15), (15, 16), (17, 20), (21, 27), (28, 31), (32, 33), (34, 42), (43, 52), (52, 53), (54, 58), (59, 62), (63, 67), (68, 76), (76, 77), (77, 78), (79, 83), (84, 88), (89, 91), (92, 93), (94, 100), (101, 107), (108, 110), (111, 114), (115, 121), (122, 126), (126, 127), (128, 139), (140, 142), (143, 148), (149, 151), (152, 155), (0, 0)]
    i: 1
    offset: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47), (48, 52), (53, 55), (56, 59), (59, 63), (64, 70), (70, 71), (0, 0), (15, 16), (17, 20), (21, 27), (28, 31), (32, 33), (34, 42), (43, 52), (52, 53), (54, 58), (59, 62), (63, 67), (68, 76), (76, 77), (77, 78), (79, 83), (84, 88), (89, 91), (92, 93), (94, 100), (101, 107), (108, 110), (111, 114), (115, 121), (122, 126), (126, 127), (128, 139), (140, 142), (143, 148), (149, 151), (152, 155), (156, 160), (161, 169), (0, 0)]
    i: 2
    offset: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47), (48, 52), (53, 55), (56, 59), (59, 63), (64, 70), (70, 71), (0, 0), (21, 27), (28, 31), (32, 33), (34, 42), (43, 52), (52, 53), (54, 58), (59, 62), (63, 67), (68, 76), (76, 77), (77, 78), (79, 83), (84, 88), (89, 91), (92, 93), (94, 100), (101, 107), (108, 110), (111, 114), (115, 121), (122, 126), (126, 127), (128, 139), (140, 142), (143, 148), (149, 151), (152, 155), (156, 160), (161, 169), (170, 173), (174, 180), (0, 0)]
    i: 3
    offset: [(0, 0), (0, 2), (3, 7), (8, 11), (12, 15), (16, 22), (23, 27), (28, 37), (38, 44), (45, 47), (48, 52), (53, 55), (56, 59), (59, 63), (64, 70), (70, 71), (0, 0), (32, 33), (34, 42), (43, 52), (52, 53), (54, 58), (59, 62), (63, 67), (68, 76), (76, 77), (77, 78), (79, 83), (84, 88), (89, 91), (92, 93), (94, 100), (101, 107), (108, 110), (111, 114), (115, 121), (122, 126), (126, 127), (128, 139), (140, 142), (143, 148), (149, 151), (152, 155), (156, 160), (161, 169), (170, 173), (174, 180), (181, 183), (183, 184), (0, 0)]


#### 3.2.7 overflow_to_sample_mapping 指明每个sample属于原始的第i个sentence？


```python
print(tokenized_examples['overflow_to_sample_mapping'])
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

