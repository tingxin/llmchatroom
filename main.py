import datetime
import json

import requests
from flask import Flask, render_template, request, session,jsonify
import os
import uuid
from LRU_cache import LRUCache
import threading
import pickle
import asyncio
import yaml

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import test_embedding
import llm
import queryllm
from random import randint
import opensearch


default_user = "tingxin"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    PORT = config['PORT']
    CHAT_CONTEXT_NUMBER_MAX = config['CHAT_CONTEXT_NUMBER_MAX']     # 连续对话模式下的上下文最大数量 n，即开启连续对话模式后，将上传本条消息以及之前你和GPT对话的n-1条消息
    USER_SAVE_MAX = config['USER_SAVE_MAX']     # 设置最多存储n个用户，当用户过多时可适当调大


PORT = os.getenv("PORT", default=PORT)  # 如果环境变量中设置了PORT，则使用环境变量中的PORT

STREAM_FLAG = False  # 是否开启流式推送
USER_DICT_FILE = "all_user_dict_v2.pkl"  # 用户信息存储文件（包含版本）
lock = threading.Lock()  # 用于线程锁

project_info = "## 西云数据客户咨询互助系统    \n" \
               "请各位同事，这里是互相技术探讨和回答客户问题的地方，请不要聊和工作不相关的话题  \n"
def get_response_from_llm(message_context):
    data = queryllm.query(message_context)
    return data


def get_message_context(message_history, have_chat_context, chat_with_history):
    """
    获取上下文
    :param message_history:
    :param have_chat_context:
    :param chat_with_history:
    :return:
    """
    message_context = []

    total = 0
    if chat_with_history:
        num = min([len(message_history), CHAT_CONTEXT_NUMBER_MAX, have_chat_context])
        # 获取所有有效聊天记录
        valid_start = 0
        valid_num = 0
        for i in range(len(message_history) - 1, -1, -1):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                valid_start = i
                valid_num += 1
            if valid_num >= num:
                break

        for i in range(valid_start, len(message_history)):
            message = message_history[i]
            if message['role'] in {'assistant', 'user'}:
                message_context.append(message)
                total += len(message['content'])
    else:
        message_context.append(message_history[-1])
        total += len(message_history[-1]['content'])

    msg = ','.join([item['content'] for item in message_history])
    query_response = test_embedding.query_endpoint(msg.encode('utf-8'))
    embedding, text_content = test_embedding.parse_response(query_response)
    text = opensearch.search(embedding, text_content)
    print(embedding)
    print(f"len(message_context): {len(message_context)} total: {total}",)
    message_context.append(text)
    return message_context


def handle_messages_get_response(message, message_history, have_chat_context, chat_with_history):
    """
    处理用户发送的消息，获取回复
    :param message: 用户发送的消息
    :param apikey:
    :param message_history: 消息历史
    :param have_chat_context: 已发送消息数量上下文(从重置为连续对话开始)
    :param chat_with_history: 是否连续对话
    """
    message_history.append({"name": default_user, "content": message})
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)


    prompts =[f"hi, 我的名字叫 {item['name']}, 这个问题我认为是这样的：\n {item['content']}" for item in message_context if item!=""]
    prompt = ','.join(prompts)

    response = get_response_from_llm(prompt)

    random_index = randint(0, len(all_users)-1)
    name = all_users[random_index]
    message_history.append({"name": name, "content": response})

    return response


def get_response_stream(message_context, message_history):
    
    gen = queryllm.query_stream(message_context)

    return gen


def handle_messages_get_response_stream(message, message_history, have_chat_context, chat_with_history):
    message_history.append({"name": "user", "content": message})
    asyncio.run(save_all_user_dict())
    message_context = get_message_context(message_history, have_chat_context, chat_with_history)
    print(f"===============================\n{message_history}")
    prompts =[f"the user's role is {item['role']}, he or she say {item['content']} to you" for item in message_context]
    prompt = ','.join(prompts)
    generate = get_response_stream(prompt, message_history)
    return generate


def check_session(current_session):
    """
    检查session，如果不存在则创建新的session
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('session_id') is not None:
        print("existing session, session_id:\t", current_session.get('session_id'))
    else:
        current_session['session_id'] = uuid.uuid1()
        print("new session, session_id:\t", current_session.get('session_id'))
    return current_session['session_id']


def check_user_bind(current_session):
    """
    检查用户是否绑定，如果没有绑定则重定向到index
    :param current_session: 当前session
    :return: 当前session
    """
    if current_session.get('user_id') is None:
        return False
    return True


def get_user_info(user_id):
    """
    获取用户信息
    :param user_id: 用户id
    :return: 用户信息
    """
    lock.acquire()
    user_info = all_user_dict.get(user_id)
    lock.release()
    return user_info


# 进入主页
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    主页
    :return: 主页
    """
    check_session(session)
    return render_template('index.html')


@app.route('/loadHistory', methods=['GET', 'POST'])
def load_messages():
    """
    加载聊天记录
    :return: 聊天记录
    """
    check_session(session)
    if session.get('user_id') is None:
        messages_history = [{"name": "assistant", "content": project_info}]
    else:
        user_info = get_user_info(session.get('user_id'))
        chat_id = user_info['selected_chat_id']
        messages_history = user_info['chats'][chat_id]['messages_history']
        print(f"用户({session.get('user_id')})加载聊天记录，共{len(messages_history)}条记录")
    return {"code": 0, "data": messages_history, "message": ""}


@app.route('/loadChats', methods=['GET', 'POST'])
def load_chats():
    """
    加载聊天联系人
    :return: 聊天联系人
    """
    check_session(session)
    if not check_user_bind(session):
        chats = []

    else:
        user_info = get_user_info(session.get('user_id'))
        chats = []
        for chat_id, chat_info in user_info['chats'].items():
            chats.append(
                {"id": chat_id, "name": chat_info['name'], "selected": chat_id == user_info['selected_chat_id']})

    return {"code": 0, "data": chats, "message": ""}


def new_chat_dict(user_id, name, send_time):
    return {"chat_with_history": False,
            "have_chat_context": 0,  # 从每次重置聊天模式后开始重置一次之后累计
            "name": name,
            "messages_history": [{"name": default_user, "content": "创建了一个新的会话"}]}


def new_user_dict(user_id, send_time):
    chat_id = str(uuid.uuid1())
    user_dict = {"chats": {chat_id: new_chat_dict(user_id, "默认对话", send_time)},
                 "selected_chat_id": chat_id,
                 "default_chat_id": chat_id}

    # user_dict['chats'][chat_id]['messages_history'].insert(1, {"name": "assistant",
    #                                                            "content": "- 创建新的用户id成功，请牢记该id  \n"
    #                                                                       })
    return user_dict


@app.route('/returnMessage', methods=['GET', 'POST'])
def return_message():
    """
    获取用户发送的消息，调用get_chat_response()获取回复，返回回复，用于更新聊天框
    :return:
    """
    print('return_message')
    check_session(session)
    send_message = request.values.get("send_message").strip()
    send_time = request.values.get("send_time").strip()
    url_redirect = "url_redirect:/"
    if send_message == "帮助":
        return "### 帮助\n" \
               "1. 输入`new:xxx`创建新的用户id\n " \
               "2. 输入`id:your_id`切换到已有用户id，新会话时无需加`id:`进入已有用户\n" \
               "3. 输入`set_apikey:`[your_apikey](https://platform.openai.com/account/api-keys)设置用户专属apikey，`set_apikey:none`可删除专属key\n" \
               "4. 输入`rename_id:xxx`可将当前用户id更改\n" \
               "5. 输入`查余额`可获得余额信息及最近几天使用量\n" \
               "6. 输入`帮助`查看帮助信息"

    if session.get('user_id') is None:  # 如果当前session未绑定用户
        print("当前会话为首次请求，用户输入:\t", send_message)
        if send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            if user_id in all_user_dict:
                session['user_id'] = user_id
                return url_redirect
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)  # 默认普通对话
            lock.release()
            print("创建新的用户id:\t", user_id)
            session['user_id'] = user_id
            return url_redirect
        else:
            user_id = send_message
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                print("已有用户id:\t", user_id)
                # 重定向到index
                return url_redirect
    else:  # 当存在用户id时
        if send_message.startswith("id:"):
            user_id = send_message.split(":")[1].strip()
            user_info = get_user_info(user_id)
            if user_info is None:
                return "用户id不存在，请重新输入或创建新的用户id"
            else:
                session['user_id'] = user_id
                print("切换到已有用户id:\t", user_id)
                # 重定向到index
                return url_redirect
        elif send_message.startswith("new:"):
            user_id = send_message.split(":")[1]
            if user_id in all_user_dict:
                return "用户id已存在，请重新输入或切换到已有用户id"
            session['user_id'] = user_id
            user_dict = new_user_dict(user_id, send_time)
            lock.acquire()
            all_user_dict.put(user_id, user_dict)
            lock.release()
            print("创建新的用户id:\t", user_id)
            return url_redirect
        elif send_message.startswith("delete:"):  # 删除用户
            user_id = send_message.split(":")[1]
            if user_id != session.get('user_id'):
                return "只能删除当前会话的用户id"
            else:
                lock.acquire()
                all_user_dict.delete(user_id)
                lock.release()
                session['user_id'] = None
                print("删除用户id:\t", user_id)
                # 异步存储all_user_dict
                asyncio.run(save_all_user_dict())
                return url_redirect
        # elif send_message.startswith("set_apikey:"):
        #     apikey = send_message.split(":")[1]
        #     user_info = get_user_info(session.get('user_id'))
        #     user_info['apikey'] = apikey
        #     print("设置用户专属apikey:\t", apikey)
        #     return "设置用户专属apikey成功"
        elif send_message.startswith("rename_id:"):
            new_user_id = send_message.split(":")[1]
            user_info = get_user_info(session.get('user_id'))
            if new_user_id in all_user_dict:
                return "用户id已存在，请重新输入"
            else:
                lock.acquire()
                all_user_dict.delete(session['user_id'])
                all_user_dict.put(new_user_id, user_info)
                lock.release()
                session['user_id'] = new_user_id
                asyncio.run(save_all_user_dict())
                print("修改用户id:\t", new_user_id)
                return f"修改成功,请牢记新的用户id为:{new_user_id}"
        else:  # 处理聊天数据
            user_id = session.get('user_id')
            print(f"用户({user_id})发送消息:{send_message}")
            user_info = get_user_info(user_id)
            chat_id = user_info['selected_chat_id']
            messages_history = user_info['chats'][chat_id]['messages_history']
            chat_with_history = user_info['chats'][chat_id]['chat_with_history']
            # apikey = user_info.get('apikey')
            # if chat_with_history:
            #     user_info['chats'][chat_id]['have_chat_context'] += 1
            # if send_time != "":
            #     messages_history.append({'name': 'system', "content": send_time})
            if not STREAM_FLAG:
                content = handle_messages_get_response(send_message, messages_history,
                                                       user_info['chats'][chat_id]['have_chat_context'],
                                                       chat_with_history)
                if send_message.startswith("@"):
                    index = send_message.find(' ')
                    name = send_message[1:index]
                else:
                    random_index = randint(0, len(all_users)-1)
                    name = all_users[random_index]
               
                return jsonify({"name": name, "content": content})

            else:
                generate = handle_messages_get_response_stream(send_message, messages_history,
                                                               user_info['chats'][chat_id]['have_chat_context'],
                                                               chat_with_history)

                if chat_with_history:
                    user_info['chats'][chat_id]['have_chat_context'] += 1
               
                return app.response_class(generate(), mimetype='application/json')


async def save_all_user_dict():
    """
    异步存储all_user_dict
    :return:
    """
    await asyncio.sleep(0)
    lock.acquire()
    with open(USER_DICT_FILE, "wb") as f:
        pickle.dump(all_user_dict, f)
    # print("all_user_dict.pkl存储成功")
    lock.release()


@app.route('/getMode', methods=['GET'])
def get_mode():
    """
    获取当前对话模式
    :return:
    """
    check_session(session)
    if not check_user_bind(session):
        return "normal"
    user_info = get_user_info(session.get('user_id'))
    chat_id = user_info['selected_chat_id']
    chat_with_history = user_info['chats'][chat_id]['chat_with_history']
    if chat_with_history:
        return {"mode": "continuous"}
    else:
        return {"mode": "normal"}


@app.route('/changeMode/<status>', methods=['GET'])
def change_mode(status):
    """
    切换对话模式
    :return:
    """
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_info = get_user_info(session.get('user_id'))
    chat_id = user_info['selected_chat_id']
    if status == "normal":
        user_info['chats'][chat_id]['chat_with_history'] = False
        print("开启普通对话")
        message = {"name": "system", "content": "切换至普通对话"}
    else:
        user_info['chats'][chat_id]['chat_with_history'] = True
        user_info['chats'][chat_id]['have_chat_context'] = 0
        print("开启连续对话")
        message = {"name": "system", "content": "切换至连续对话"}
    user_info['chats'][chat_id]['messages_history'].append(message)
    return {"code": 200, "data": message}


@app.route('/selectChat', methods=['GET'])
def select_chat():
    """
    选择聊天对象
    :return:
    """
    chat_id = request.args.get("id")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    user_info['selected_chat_id'] = chat_id
    return {"code": 200, "msg": "选择聊天对象成功"}


@app.route('/newChat', methods=['GET'])
def new_chat():
    """
    新建聊天对象
    :return:
    """
    name = request.args.get("name")
    time = request.args.get("time")
    check_session(session)
    if not check_user_bind(session):
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_id = session.get('user_id')
    user_info = get_user_info(user_id)
    new_chat_id = str(uuid.uuid1())
    user_info['selected_chat_id'] = new_chat_id
    user_info['chats'][new_chat_id] = new_chat_dict(user_id, name, time)
    print("新建聊天对象")
    return {"code": 200, "data": {"name": name, "id": new_chat_id, "selected": True}}


@app.route('/deleteHistory', methods=['GET'])
def delete_history():
    """
    清空上下文
    :return:
    """
    check_session(session)
    if not check_user_bind(session):
        print("请先创建或输入已有用户id")
        return {"code": -1, "msg": "请先创建或输入已有用户id"}
    user_info = get_user_info(session.get('user_id'))
    chat_id = user_info['selected_chat_id']
    default_chat_id = user_info['default_chat_id']
    if default_chat_id == chat_id:
        print("清空历史记录")
        user_info["chats"][chat_id]['messages_history'] = user_info["chats"][chat_id]['messages_history'][:5]
    else:
        print("删除聊天对话")
        del user_info["chats"][chat_id]
    user_info['selected_chat_id'] = default_chat_id
    return "2"

def init_users():
    with open(f"{os.getcwd()}/users.json") as f:
        users = json.load(f)
    
    return [ user['name'] for user in users]

if __name__ == '__main__':
    # print("持久化存储文件路径为:", os.path.join(os.getcwd(), USER_DICT_FILE))
    
    all_user_dict = LRUCache(USER_SAVE_MAX)
    all_users = init_users()

    app.run(host="0.0.0.0", port=5018, debug=False)