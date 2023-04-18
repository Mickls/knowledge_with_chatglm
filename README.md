# KnowledgeWithChatGLM
## 介绍
KnowledgeWithChatGLM是一个基于向量索引库[Milvus](https://github.com/milvus-io/milvus)和[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)实现的知识库问答应用

## 运行环境
- 该项目默认基于GPU启动运行，需要安装[nvidia](https://www.nvidia.cn/geforce/drivers/)驱动和[cuda](https://developer.nvidia.com/cuda-toolkit-archive)驱动
- python 3.8+
- milvus

## 检查pytorch是否支持GPU
```python
import torch

print(torch.cuda.is_available())
```

## 使用方式
1. `pip install -r requirements.txt`
2. 部署`milvus`服务: `docker-compose up -d`
3. 初始化向量索引库，将文档预处理并转为向量建立索引: `python document_preprocess.py`
4. 启动服务: `python web_demo.py`

如果对本项目源码有更多的好奇，可以参考以下文章[chatglm实现基于知识库问答的应用](https://blog.csdn.net/a914541185/article/details/130150101)