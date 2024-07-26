import gradio as gr
import os

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


def initialize_sales_bot(vector_store_dir: str="sales_query"): 
    os.environ['OPENAI_API_KEY'] = 'sk-LGbJjRIZZ6VB4Zi8C0Fc3e57F3Bd45838276F801De5e88Ef'
    os.environ['OPENAI_BASE_URL'] = 'https://api.xiaoai.plus/v1'
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    #当User的问题没有包含在向量数据库中时
    if len(ans["source_documents"]) == 0:
        prompt_template = """
                对话内容历史为：{history}
                User的新问题是：{question}
                像一个专业的护肤品销售员一样给出一个合理又不让客户失望的回答
                """
        llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5)

        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question"])
        #使用上面的模版和llm对象创建LLMChain对象
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(history=history, question=message)
        print(f"[template response] {response}")
        return response
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="护肤品销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
