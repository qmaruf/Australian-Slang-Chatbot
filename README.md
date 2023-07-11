---
title: AustralianSlangChatbot
emoji: 💬
colorFrom: '#FDB157'
colorTo: '#FFA773'
app_file: aussie_bot.py
sdk: gradio
sdk_version: 3.0.11
pinned: true
---
# Australian Slang Chatbot

This is a chatbot that answers using Australian slang. The chatbot, named "Brissy," is powered by OpenAI's language model and can provide responses in a fair dinkum Aussie manner. It's designed to engage in conversations about Australia, Aussie culture, and a wide range of topics while incorporating friendly and authentic Australian slang.

## Dependencies
The following dependencies are required to run the chatbot:

* gradio
* openai
* langchain

To install the dependencies, you can use the following command:

```bash
pip install gradio openai-python langchain
```
Please ensure you have the appropriate access credentials for OpenAI's API by setting the `OPENAI_API_KEY` environment variable.

## Usage
To use the chatbot, follow these steps:

1. Import the required modules and libraries.
2. Set up the get_template() function, which returns the template for the chatbot's responses. It includes a placeholder for history and human input.
3. Implement the get_chain() function, which returns the chatbot chain configuration using the LLMChain from the langchain library. This includes setting up the language model, prompt template, verbosity, and conversation memory.
4. Define the interface() function, which launches the chatbot interface using the gradio library. It sets up the chatbot, message textbox, and clear button. It also includes the logic for user input and bot responses.
5. Execute the interface() function to start the chatbot interface.

## Notes
1. The chatbot utilizes the OpenAI language model with a temperature setting of 0.5, which controls the randomness of the generated responses.
2. The ConversationBufferWindowMemory is used to store and manage conversation history, with a buffer size of 100.
3. The chatbot interface is created using the gradio library, providing an interactive and user-friendly experience.
