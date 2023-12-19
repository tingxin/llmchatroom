# 使用大预言模型构建咨询系统
## 准备数据
```bash
export EFS_DIR=/home/ec2-user/datasets
```

```bash
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.aws.amazon.com --no-parent --accept=html \
  -P $EFS_DIR https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html
```

 ## 前提条件
 1. 在sagemaker上使用jumpstart部署一个text embedding 模型
 2. 在sagemaker上使用jumpstart部署一个随意部署一个text generative模型，模型约大，精度越好
 3. 部署一个opensarch作为向量数据库，使用opensearch.py的代码进行数据的预先准备

## 部署方法
前提：python3.7及以上运行环境
1. 执行 `pip install -r requirements.txt`安装必要包
2. 打开`config.yaml`文件
5. 执行`python main.py`运行程序.
6. 打开本地浏览器访问`127.0.0.1:5018`,部署完成
