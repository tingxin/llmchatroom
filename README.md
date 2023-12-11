# 使用大预言模型构建咨询系统
## 准备数据
```
We need to first download the [Ray documentation](https://docs.ray.io/) to a directory:
```bash
export EFS_DIR=/home/ec2-user/datasets
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.aws.amazon.com --no-parent --accept=html \
  -P $EFS_DIR https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html
```
```

 

## 部署方法
分别介绍下面几种部署方法，选择一种即可，部署完成后直接跳转至后面的使用介绍继续即可
<details>
<summary>1. 本地源代码部署（推荐，方便更新，需要有代理）</summary>

> 前提：python3.7及以上运行环境
> 1. 执行 `pip install -r requirements.txt`安装必要包
> 2. 打开`config.yaml`文件
> 5. 执行`python main.py`运行程序.
> 6. 打开本地浏览器访问`127.0.0.1:5000`,部署完成
> 7. 关于更新，当代码更新时，使用git pull更新重新部署即可  
</details>
<details>
<summary>2. Railway部署（推荐，无需代理，云部署，通过url随时随地访问）</summary>  
  
 
  > . 将会跳转至新页面，依次添加`PORT`,`DEPLOY_ON_RAILWAY`以及`OPENAI_API_KEY`三个环境变量,相应值如下PORT为5000，DEPLOY_ON_RAILWAY为true




 
  
</details>


 # 方案
 基于langchain构造本地数据知识库，前后端可以直接部署服务