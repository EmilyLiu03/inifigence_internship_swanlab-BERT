import wandb
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import os

# 初始化WandB项目
wandb.init(project="bert-imdb", entity="emilylmh03-blcu")

# 设置WandB配置
config = wandb.config
config.learning_rate = 2e-5
config.num_train_epochs = 3
config.per_device_train_batch_size = 8
config.per_device_eval_batch_size = 8

# 定义加载本地IMDB数据集的函数
def load_imdb_dataset(data_dir):
    train_data = []
    test_data = []

    # 读取训练数据
    for label in ['pos', 'neg']:
        folder_path = os.path.join(data_dir, 'train', label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    train_data.append({'text': file.read(), 'label': 1 if label == 'pos' else 0})

    # 读取测试数据
    for label in ['pos', 'neg']:
        folder_path = os.path.join(data_dir, 'test', label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                    test_data.append({'text': file.read(), 'label': 1 if label == 'pos' else 0})

    return DatasetDict({
        'train': Dataset.from_pandas(pd.DataFrame(train_data)),
        'test': Dataset.from_pandas(pd.DataFrame(test_data))
    })

# 加载本地IMDB数据集
data_dir = '/root/aclImdb'  # 替换为您的IMDB数据集解压后的路径
dataset = load_imdb_dataset(data_dir)

# 加载预训练的BERT tokenizer和模型
model_path = '/root/bert-base-uncased'  # 替换为您的BERT模型文件路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 定义tokenize函数
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

# 对数据集进行tokenization
tokenized_datasets = dataset.map(tokenize, batched=True)

# 设置模型输入格式
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    learning_rate=config.learning_rate,
    logging_dir='./logs',  # 指定日志目录
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')

# 测试模型
test_reviews = [
    "I absolutely loved this movie! The storyline was captivating and the acting was top-notch. A must-watch for everyone.",
    "This movie was a complete waste of time. The plot was predictable and the characters were poorly developed.",
    "An excellent film with a heartwarming story. The performances were outstanding, especially the lead actor.",
    "I found the movie to be quite boring. It dragged on and didn't really go anywhere. Not recommended.",
    "A masterpiece! The director did an amazing job bringing this story to life. The visuals were stunning.",
    "Terrible movie. The script was awful and the acting was even worse. I can't believe I sat through the whole thing.",
    "A delightful film with a perfect mix of humor and drama. The cast was great and the dialogue was witty.",
    "I was very disappointed with this movie. It had so much potential, but it just fell flat. The ending was particularly bad.",
    "One of the best movies I've seen this year. The story was original and the performances were incredibly moving.",
    "I didn't enjoy this movie at all. It was confusing and the pacing was off. Definitely not worth watching."
]

model.to('cpu')
text_list = []
CLASS_NAME = {0: "negative", 1: "positive"}
for review in test_reviews:
    inputs = tokenizer(review, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    text_list.append({"text": review, "label": predicted_class, "class_name": CLASS_NAME[predicted_class]})

if text_list:
    wandb.log({"predictions": text_list})