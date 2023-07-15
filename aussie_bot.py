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
    template = """Brissy is an Australian Slang Chatbot based on large language model.

    Brissy is a fair dinkum Aussie model and knows all about Australian slang. It's a top-notch mate and can answer questions about Australia, Aussie culture, and a whole bunch of other topics. It always uses friendly slang and can chat like a true blue Aussie. Brissy start answering every question differently. Brissy will always answer every question within 4000 characters.

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
        memory=ConversationBufferWindowMemory(k=5),
    )
    return chatgpt_chain


def buy_me_a_coffee() -> str:
    """
    Returns the buy me a coffee button
    """
    return """
        <p style="margin-bottom: 10px; font-size: 60%">
        <span style="display: flex;align-items: center;justify-content: center;height: 30px;">
        <a href="https://www.buymeacoffee.com/qmaruf">
        <img src="https://badgen.net/badge/icon/Buy%20Me%20A%20Coffee?icon=buymeacoffee&label" alt="Buy me a coffee"></a>
        </span>
        </p>
    """


def interface() -> None:
    """
    Launches the chatbot interface.
    """
    with gr.Blocks() as demo:
        gr.HTML(buy_me_a_coffee())
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button('Clear')

        try:
            chatgpt_chain = get_chain()
        except Exception as e:
            print(e)
            chatgpt_chain = None

        def user(user_message, history):
            if len(history) > 3500:
                history = history[-3500:]
            return '', history + [[user_message, None]]

        def bot(history):
            try:
                human_input = history[-1][0]

                if chatgpt_chain is None:
                    raise Exception('Chatbot not initialized')

                if len(human_input) < 512:
                    response = chatgpt_chain.predict(human_input=human_input)
                else:
                    response = 'Sorry, I can only answer questions shorter than 512 characters.'
            except Exception as e:
                print(e)
                response = 'Sorry, I had trouble answering that question. Please try again.'

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
