"""
# Depreciated pipelien due to no longer model transfroemr support pipeline for translation on huggingface
from transformers import pipeline

class HFTranslator:
    def __init__(self):
        # NLLB-200 is multilingual and auto-detects source languages well
        # 600M is small enough for CPU inference
        self.translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")

    def translate_to_english(self, texts):
        # Force the output to English (eng_Latn)
        results = self.translator(texts, tgt_lang="eng_Latn", max_length=512)
        return [res['translation_text'] for res in results]
"""

"""
# poor transaltion 

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HFTranslator:
    def __init__(self):
        model_id = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        # NLLB requires you to explicitly map the target language token
        self.eng_lang_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate_to_english(self, texts):
        # Tokenize the batch of reviews
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Generate translations manually, forcing the first output token to be English
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.eng_lang_id,
                max_new_tokens=512
            )
            
        # Decode the generated tokens back to readable strings
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

"""
"""
# CPU GPU device update
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langdetect import detect, LangDetectException

class HFTranslator:
    def __init__(self, model_id="Helsinki-NLP/opus-mt-it-en"): # "facebook/nllb-200-1.3B"):
        # Load model here
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,  # half precision ofr speed only
            device_map="cuda:0"
        )
        # self.eng_lang_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate_to_english(self, texts):
        results = [""] * len(texts)
        to_translate_indices = []
        to_translate_texts = []
        
        # Step 1: Fast CPU check for English
        for i, text in enumerate(texts):
            try:
                if detect(text) == 'en':
                    results[i] = text # Keep original English text
                else:
                    to_translate_indices.append(i)
                    to_translate_texts.append(text)
            except LangDetectException:
                # Triggers if the text has no letters (e.g., just "10/10" or emojis)
                results[i] = text 

        # If everything was already English, skip the GPU entirely
        if not to_translate_texts:
            return results
            
        # Step 2: Batch translate ONLY the non-English text
        inputs = self.tokenizer(
            to_translate_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to("cuda:0") 
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                # forced_bos_token_id=self.eng_lang_id,
                max_new_tokens=512
            )
            
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Step 3: Insert translations back into their original positions
        for idx, translated_text in zip(to_translate_indices, translated):
            results[idx] = translated_text
            
        return results
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from langdetect import detect, LangDetectException

class HFTranslator:
    def __init__(self, model_id="Helsinki-NLP/opus-mt-it-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 1. Dynamically detect the best available hardware
        if torch.cuda.is_available():
            self.device = "cuda:0"  # Nvidia GPUs (Kaggle)
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"     # Apple Silicon Macs (M1/M2/M3)
            self.dtype = torch.float16
        else:
            self.device = "cpu"     # Older Macs or basic servers
            self.dtype = torch.float32 # CPU usually requires float32

        # 2. Load the model onto the detected device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)

    def translate_to_english(self, texts):
        results = [""] * len(texts)
        to_translate_indices = []
        to_translate_texts = []
        
        for i, text in enumerate(texts):
            try:
                if detect(text) == 'en':
                    results[i] = text
                else:
                    to_translate_indices.append(i)
                    to_translate_texts.append(text)
            except LangDetectException:
                results[i] = text 

        if not to_translate_texts:
            return results
            
        # 3. Send inputs to the dynamically detected device
        inputs = self.tokenizer(
            to_translate_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512
            )
            
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for idx, translated_text in zip(to_translate_indices, translated):
            results[idx] = translated_text
            
        return results