from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech,SpeechT5HifiGan
from datasets import load_dataset
import torch
from playsound import playsound
from asr_llm import ASR_LLM
import scipy

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speaker_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")


if __name__ == "__main__":
    main_class=ASR_LLM('base','Mistral-Nemo-Instruct-2407')
    asr_model=ASR_LLM.asr_model_load()
    playsound('greetings.wav')
    input_speech=ASR_LLM.asr_call(asr_model)
    llm_response=ASR_LLM.llm_call(input_speech)
    inputs = processor(text=llm_response, return_tensors="pt")
    spectrogram_final = model.generate_speech(inputs["input_ids"], speaker_embeddings,vocoder=vocoder)
    scipy.io.wavfile.write("final_response.wav", rate=speaker_model.config.sampling_rate, data=output)
    playsound("final_response.wav")
    



    
    