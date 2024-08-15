from langchain_community.llms import VLLMOpenAI
from langchain_community.utilities import SerpAPIWrapper
import os
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
import speech_recognition as sr
import io
import soundfile as sf
import numpy as np
import whisper
from prompt_template import system, human
#search = SerpAPIWrapper()


class ASR_LLM:
    def __init__(self,asr_model,llm_name):
        self.asr_model=asr_model
        self.llm_name=llm_name

    def llm_call(self,query):
        llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    model_name=self.llm_name,
    max_tokens=2048)

        prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)       #wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2500))
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Never search for more than one concept at a single step. If you need to compare two concepts, search for each one individually. Syntax: string with a simple concept",
            func=wikipedia.run
        )
        agent = create_json_chat_agent(
        tools = [wikipedia_tool],
        llm = llm,
        prompt = prompt,
        stop_sequence = ["STOP"],
        template_tool_response = "{observation}")

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent_executor = AgentExecutor(agent=agent, tools=[wikipedia_tool], verbose=True, handle_parsing_errors=True, memory=memory)
        output=agent_executor.invoke({"input": query})
        
        return output['output']
    def asr_model_load(self):
        r = sr.Recognizer()
        asr_model=whisper.load_model('base')
        return asr_model

    def asr_call(self,asr_module):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
        wav_bytes = audio.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)
        result=asr_module.transcribe(audio_array,language='english',fp16=True)
        return result['text']


        
