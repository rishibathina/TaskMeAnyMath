from typing import Callable, Union

from ... import constant
import torch

# import openai
import transformers
import re

from .base_qa_model import QAModel, QAModelInstance

textqa_models = {
    "Meta-Llama-3-8B-Instruct": ("HuggingFace", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "gemma-2-9b-it": ("HuggingFace", "google/gemma-2-9b-it"),
    "Qwen2-7B-Instruct": ("HuggingFace", "Qwen/Qwen2-7B-Instruct"),
    # olmo-7b-instruct
    # gpt4o-mini
    # gpt4o
    # claude-3.5-sonnet
}


def set_textqa_model_key(model_name, key):
    textqa_models[model_name] = (textqa_models[model_name][0], key)


def list_textqa_models():
    return list(textqa_models.keys())


class TextQAModel(QAModel):
    def __init__(
        self,
        model_name: str,
        prompt_name: str,
        prompt_func: Callable,
        model: QAModelInstance = None,
        torch_device: Union[int, str] = -1,
        precision=torch.bfloat16,
        choice_format="letter",
        enable_choice_search: bool = False,
        cache_path: str = None,
    ):
        super().__init__(
            model_name,
            prompt_name,
            prompt_func,
            choice_format,
            enable_choice_search,
            cache_path,
        )
        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        else:
            if torch_device == -1:
                torch_device = (
                    torch.device("cuda") if torch.cuda.is_available() else "cpu"
                )
            else:
                torch_device = torch.device(f"cuda:{torch_device}")

        if model is None:
            print(f"Loading {model_name}...")
            class_name, ckpt = textqa_models[model_name]
            print(class_name, ckpt)
            self.model_precision = precision
            self.model = eval(class_name)(ckpt, torch_device, self.model_precision)
            print(f"Finish loading {model_name}")
        else:
            print(f"Using provided model...")
            self.model = model

    def _data_to_str(self, data):
        assert isinstance(data, str)
        return data


class HuggingFace(QAModelInstance):
    def __init__(
        self, ckpt, torch_device=torch.device("cuda"), precision=torch.float32
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, device_map=torch_device, torch_dtype=precision
        )
        # device_map= device_map="auto"
        
        # self.model = transformers.pipeline(
        #     "text-generation",
        #     model=ckpt,
        #     device=torch_device,
        #     max_new_tokens=constant.MAX_OUTPUT_LEN,
        #     return_full_text=False,
        # )

    def qa(self, data, prompt):
        outputs = self.model.generate(
            **self.tokenizer(prompt, return_tensors="pt"),
            max_new_tokens=constant.MAX_OUTPUT_LEN,
            do_sample=False,
            top_p=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # return self.model(prompt)[0]["generated_text"]


class MathQAModel(QAModel):
    @torch.no_grad()
    def qa(self, data, question, answer=None):
        prompt = self.prompt_func(question)
        free_form_answer = self._qa(data, prompt)

        result = {"free_form_answer": free_form_answer}

        if answer is not None:
            pattern = r"\{([^}]*)\}"
            match = re.search(pattern, free_form_answer)
            if match:
                result["answer"] = match.group(1)
        return result
