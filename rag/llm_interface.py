import os, sys
import logging
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class LLMInterface:
    def __init__(self) -> None:
        self.model_name = None

    def query(self, system: str, user_message: str):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

def get_llm_interface(llm_type: str):
    """
    Factory function to get an LLM interface based on the type specified.

    Parameters
    ----------
    llm_type : str
        The type of the large language model, either "openai" or "huggingface".

    Returns
    -------
    LLMInterface
        An instance of a subclass of LLMInterface corresponding to the given llm_type.

    Raises
    ------
    NotImplementedError
        If the llm_type is not recognized.
    """
    if llm_type == "openai":
        class OpenAIInterface(LLMInterface):
            def __init__(self) -> None:
                super().__init__()
                self.model_name = os.getenv("LLM_NAME", default="gpt-3.5-turbo")
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if openai.api_key is None:
                    logging.warning("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

            def query(self, system: str, user_message: str) -> str:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.choices[0].message.content

        return OpenAIInterface()
    elif llm_type == "huggingface":
        class HuggingFaceInterface(LLMInterface):
            def __init__(self) -> None:
                super().__init__()
                model_name = os.getenv("LLM_NAME")
                cache_dir = os.getenv("CACHE_DIR")
                if not os.path.isdir(cache_dir):
                    raise ValueError(f"Specified cache_dir {cache_dir} is not a directory")

                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    rope_scaling={"type": "dynamic", "factor": 2},
                    cache_dir=cache_dir,
                )
                self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            def query(self, instruction: str, user_message: str) -> str:
                inputs = self.format_input(instruction, user_message)
                output = self.model.generate(**inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
                response = self.tokenizer.decode(output[0], skip_special_tokens=True).split("Response")[1]
                return response.strip()

            def format_input(self, instruction: str, user_input: str) -> dict:
                prompt = f"### Instruction:\n{instruction}\n### Input:\n{user_input}\n### Response:\n"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                del inputs["token_type_ids"]
                return inputs

        return HuggingFaceInterface()

    else:
        raise NotImplementedError(f"LLM type '{llm_type}' is not implemented")

if __name__ == '__main__':
    interface = get_llm_interface(sys.argv[1])
    print(interface.query("Example system message", "Hello, world!"))