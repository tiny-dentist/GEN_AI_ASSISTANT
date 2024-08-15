# GEN_AI_ASSISTANT

# Introduction
With the introduction of Transformer architecture from the research paper "Attention Is All You Need" from 2017, there has been a rapid advancement in the field of Natural Language Processing. We now have advanced state of the art models like GPT-4o, Claude 3 and Grok which have good reasoning capability and are capable of solving complex task without requiring human intervention, ensuring they become a force multiplier in our daily parts of life. With this, Il like to demonstrate my solution called Empower AI, a speech to speech pipleline consisting of an ASR Model to convert speech to text, an LLM to take in the converted speech to text as input for solving user query's and an TTS model to convert the text back to an audio response. This solution can have huge benefits for visually challenged people, who can easily interact with the solution with their voice alone instead of relying on braille script which is hard to maintain and expensive to produce.

# Architecture and Working
Our pipeline consists of three models, OpenAI Whisper V3 model responsible for converting speech to text, Mistral Nemo 12B LLM MistralAI/Nvidia which will take the user query as input and generate an appropriate response to help the user in their task and a Text-To-Speech model responsible for converting the generated response back to an audio format that could be easily understood by the user. The LLM is also connected with an tool that allows it search for answers via the Internet incase its not able to answer the question from its internal knowledge base or requires real-time updates. The ASR model is loaded onto the GPU Memory(AMD W7900) and transcribes the audio.

The Python speech_recognition library is used to detect if a person is speaking and then captures that information and send it to the ASR model. The ASR model's output is then fed into the LLM. The LLM is hosted on the GPU via VLLM, which allows it to be easily interactable for any developer with plug-and-play capability(via OpenAI API wrapper) while using it's advanced Paged Attention implementation to ensure it efficiently manages its KV Cache and handle longer sequences of input and output. Depending on the input, the LLM decides whether or not a tool call needs to be made to help in answering the question via its internal Chain-Of-Thought Reasoning. It will give an an output in JSON format which details the steps needed to answer the question while also deciding which tool if needed along with it's argument. We define the tools via Langchain, which provides us an higer level API to create our custom finctions.  The generated output is then passed onto the tools, whose output is then passed back to the LLM to generate the final response. Finally the TTS model converts the text to an audio speech, which helps the user in easily understanding the LLM's response. 

Given below is the higher level architecture diagram for the entire pipeline:

![arch_diag](https://github.com/user-attachments/assets/73fc25c3-17a7-4e39-ac24-56c350e4947d)

# Software and Hardware Used

#### Software: Huggingface,VLLM,Whisper library, speech_recognition,Pytorch, ROCM
#### Hardware: AMD Radeon™ PRO W7900 Professional Graphics, AMD® Ryzen 5 7600x, 32GB DDR5 RAM, 1 TB SSD 
