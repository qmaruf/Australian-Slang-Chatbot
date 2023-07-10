"""
A chatbot that will answer using australian slang
"""
import os
import time

import gradio as gr
import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_template() -> str:
    """
    Returns the template for the chatbot
    """
    template = """Brissy is a large language model trained by OpenAI.

    Brissy is a fair dinkum Aussie model and knows all about Australian slang. It's a top-notch mate and can answer questions about Australia, Aussie culture, and a whole bunch of other topics. It always uses friendly slang and can chat like a true blue Aussie. Brissy start answering every question differently.

    Reckon you can rewrite your response using Australian slang?

    {history}
    Human: {human_input}
    Brissy:"""

    return template


def get_chain() -> LLMChain:
    """
    Returns the chatbot chain
    """
    template = get_template()

    prompt = PromptTemplate(
        input_variables=['history', 'human_input'],
        template=template
    )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1.0)

    chatgpt_chain = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=100),
    )
    return chatgpt_chain


def interface() -> None:
    """
    Launches the chatbot interface.
    """
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button('Clear')
        chatgpt_chain = get_chain()

        def user(user_message, history):
            return '', history + [[user_message, None]]

        def bot(history):
            human_input = history[-1][0]
            response = chatgpt_chain.predict(human_input=human_input)

            history[-1][1] = ''
            for character in response:
                history[-1][1] += character
                time.sleep(0.01)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    interface()
