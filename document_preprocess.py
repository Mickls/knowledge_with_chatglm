import os
import re
import jieba
import torch
import pandas as pd
from pymilvus import utility
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, DataType
from transformers import AutoTokenizer, AutoModel

connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)

# 定义集合名称和维度
collection_name = "document"
dimension = 768
docs_folder = "./knowledge/"

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")


# 获取文本的向量
def get_vector(text):
    input_ids = tokenizer(text, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output = model(input_ids)[0][:, 0, :].numpy()
    return output.tolist()[0]


def create_collection():
    # 定义集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="primary id"),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]

    # 定义集合模式
    schema = CollectionSchema(fields=fields, description="collection schema")

    # 创建集合

    if utility.has_collection(collection_name):
        # return
        utility.drop_collection(collection_name)
        collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
        # 创建索引
        default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 2048}, "metric_type": "IP"}
        collection.create_index(field_name="vector", index_params=default_index)
        print(f"Collection {collection_name} created successfully")
    else:
        collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
        # 创建索引
        default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 2048}, "metric_type": "IP"}
        collection.create_index(field_name="vector", index_params=default_index)
        print(f"Collection {collection_name} created successfully")


def init_knowledge():
    collection = Collection(collection_name)
    # 遍历指定目录下的所有文件，并导入到 Milvus 集合中
    docs = []
    for root, dirs, files in os.walk(docs_folder):
        for file in files:
            # 只处理以 .txt 结尾的文本文件
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 对文本进行清洗处理
                content = re.sub(r"\s+", " ", content)
                title = os.path.splitext(file)[0]
                # 分词
                words = jieba.lcut(content)
                # 将分词后的文本重新拼接成字符串
                content = " ".join(words)
                # 获取文本向量
                vector = get_vector(title + content)
                docs.append({"title": title, "content": content, "vector": vector})

    # 将文本内容和向量通过 DataFrame 一起导入集合中
    df = pd.DataFrame(docs)
    collection.insert(df)
    print("Documents inserted successfully")


if __name__ == "__main__":
    create_collection()
    init_knowledge()
