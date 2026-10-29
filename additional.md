### As I implemented multiple LLLM stages, I also need to make it run a bit faster not completely sequentially..
### Need to modify parts to use VllM
### Use of Half precision :)

### For translator
self.model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda:0"
)

### For the Sentiment/Zero-Shot pipelines in hf.py
pipeline(
    task="text-classification", 
    model=model_id, 
    torch_dtype=torch.float16, 
    device="cuda:0"
)


##########
For LLM Aspect Extraction

!pip install vllm

from vllm import LLM, SamplingParams

class VLLMChatGenerator:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        # Load onto the second GPU to distribute the load
        self.llm = LLM(model=model_id, tensor_parallel_size=1, dtype="half")
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    def generate(self, prompts):
        # vLLM handles batching automatically and incredibly fast
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

