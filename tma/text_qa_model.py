from typing import Callable, Union

import constant
import torch

# import openai
import transformers
import re
import os

from base_qa_model import QAModel, QAModelInstance

textqa_models = {
    "Meta-Llama-3.1-8B-Instruct": ("HuggingFace", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "gemma-2-9b-it": ("HuggingFace", "google/gemma-2-9b-it"),
    "Qwen2-7B-Instruct": ("HuggingFace", "Qwen/Qwen2-7B-Instruct"),
    "OLMo-7B-0424-hf": ("HuggingFace", "allenai/OLMo-7B-0424-hf"),
    "gpt-4o-mini": ("GPT", "<OpenAI_API>"),
    "claude-3-sonnet-20240229": ("Claude", "<ANTHROPIC_API>")
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


class GPT(QAModelInstance):
    def __init__(self, model_name):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def qa(self, prompt, answer_only, mc):
        max_tokens = constant.MAX_OUTPUT_LEN_REASONING
        temperature = constant.TEMPERATURE_FOR_REASONING
        top_p = constant.TOP_P_FOR_REASONING
        if answer_only:
            max_tokens = constant.MAX_OUTPUT_LEN_ANSWER_ONLY
            temperature = constant.TEMPERATURE_FOR_ANSWER_ONLY
            top_p = constant.TOP_P_FOR_ANSWER_ONLY
            
        if mc:
            max_tokens = constant.MAX_OUTPUT_LEN_MC
            
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content


class Claude(QAModelInstance):
    def __init__(self, model_name):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def qa(self, prompt, answer_only, mc):
        max_tokens = constant.MAX_OUTPUT_LEN_REASONING
        temperature = constant.TEMPERATURE_FOR_REASONING
        top_p = constant.TOP_P_FOR_REASONING
        if answer_only:
            max_tokens = constant.MAX_OUTPUT_LEN_ANSWER_ONLY
            temperature = constant.TEMPERATURE_FOR_ANSWER_ONLY
            top_p = constant.TOP_P_FOR_ANSWER_ONLY
            
        if mc:
            max_tokens = constant.MAX_OUTPUT_LEN_MC
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.context[0].text

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
    

    def qa(self, prompt, answer_only, mc):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs['input_ids'].shape[1]
        
        max_tokens = constant.MAX_OUTPUT_LEN_REASONING
        temperature = constant.TEMPERATURE_FOR_REASONING
        top_p = constant.TOP_P_FOR_REASONING
        if answer_only:
            max_tokens = constant.MAX_OUTPUT_LEN_ANSWER_ONLY
            temperature = constant.TEMPERATURE_FOR_ANSWER_ONLY
            top_p = constant.TOP_P_FOR_ANSWER_ONLY
        
        if mc:
            max_tokens = constant.MAX_OUTPUT_LEN_MC
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # remove prompt from generated tokens
        generated_tokens = outputs[0, input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
