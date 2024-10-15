import random
import torch
import re
import json
import os
import numpy as np
from openai import OpenAI
from tqdm import tqdm

import prompt
from metadata import MathTemplateMetaData
from task_store import TaskStore
from constant import template_to_generator, template_to_task_type
from text_qa_model import GPT, TextQAModel

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# all_templates = ['triangle_area_templates.json', 'volume_rectangular_prism_templates.json', 'circle_templates.json', 'perimeter_templates.json', 'sideLength_template.json', 'midpoint_templates.json', 'cone_volume_templates.json', 'angle_templates.json', 'arcLength_template.json', 'point_distance_template.json', 'pythagoreanTheorem_template.json', 'sphere_volume_templates.json', 'angle_sum_templates.json', 'perpendicular_template.json', 'intersection_templates.json']

# llama = TextQAModel(model_name='Meta-Llama-3.1-8B-Instruct', precision=torch.bfloat16,
                    # prompt_name="succinct_prompt", prompt_func=prompt.succinct_prompt, cache_path=".cache/")
gemma = TextQAModel(model_name='gemma-2-9b-it', precision=torch.bfloat16, prompt_name = "succinct_prompt", prompt_func=prompt.succinct_prompt, cache_path = ".cache/")
# qwen = TextQAModel(model_name='Qwen2-7B-Instruct', precision=torch.bfloat16, prompt_name = "succinct_prompt", prompt_func=prompt.succinct_prompt, cache_path = ".cache/")
# olmo = TextQAModel(model_name='OLMo-7B-0424-hf', precision=torch.bfloat16, prompt_name = "succinct_prompt", prompt_func=prompt.succinct_prompt, cache_path = ".cache/")
gpt4o_mini = GPT(model_name='gpt-4o-mini')
gpt4omini = TextQAModel(model_name='gpt-4o-mini', model=gpt4o_mini, precision=torch.bfloat16,
                        prompt_name="succinct_prompt", prompt_func=prompt.succinct_prompt)
# models = [llama, gemma, qwen]


def qa(model, question, options: dict):
    if not options:
        prompt = question + \
            " Output your final answer in the following format: {<final_answer>}."
        anwer_only_prompt = prompt + \
            " Only output the final answer. Don't include any explanation or reasoning."
        code_prompt = "Write the minimal python code to answer the following question: " + \
            question + \
            " Output your final answer in the following format: {<final_answer>}"
        reasoning = model.qa(prompt, answer_only=False, mc=False)
        answer_only = model.qa(anwer_only_prompt, answer_only=True, mc=False)
        # code = model.qa(code_prompt, answer_only=False, mc=False)
        # return {"answer_only": answer_only, "reasoning": reasoning, "code": code}
        return {"answer_only": answer_only, "reasoning": reasoning}
    else:
        options = list(options)
        np.random.shuffle(options)
        options = '\n'.join(options)
        prompt = (f"{question}\n" f"Select the best choice from the following options:\n"f"{options}\n" +
                  "Only output your answer in \{\} nothing else.")
        answer = model.qa(prompt, answer_only=True, mc=True)
        return {"answer": answer}
        # options_prompt = question + " " + options + " Output your final answer in the following format: {<final_answer>}."
        # options_answer_only_prompt = options_prompt + " Only output the final answer. Don't include any explanation or reasoning."
        # options_code_prompt = "Write the minimal python code to answer the following question: " + question + " Output your final answer in the following format: {<final_answer>}"
        # options_reasoning = model.qa(options_prompt, answer_only=False)
        # options_answer_only = model.qa(options_answer_only_prompt, answer_only=True)
        # options_code = model.qa(options_code_prompt, answer_only=False)
        # return {"answer_only": options_answer_only, "reasoning": options_reasoning, "code": options_code}


def generate_output(model, model_name, num_tasks, math_field, templates=None, mc=False):
    if not templates:
        math_field = "algebra"
        # math_field = "geometry"
        if math_field == "geometry":
            templates = [f for f in os.listdir("../math_annotations/geometry") if os.path.isfile(os.path.join("../math_annotations/geometry", f))]
            templates.remove("cone_volume_templates.json")
            templates.remove("perpendicular_template.json")
            templates.remove("angle_templates.json")
            
            # TODO: Add edge case to handle the below (don't strip commas)
            templates.remove("midpoint_templates.json")
            templates.remove("intersection_templates.json")
            
            # TODO: something wrong with constructor
            templates.remove("cone_volume_templates.json")

        elif math_field == "algebra":
            templates = [f for f in os.listdir("../math_annotations/algebra") if os.path.isfile(os.path.join("../math_annotations/algebra", f))]
        else:
            raise ValueError("Invalid math field")
            
    for template in tqdm(templates):
        template_path = "../math_annotations/" + math_field + "/" + template
        metadata = MathTemplateMetaData(template_path=template_path)
        generator = template_to_generator[template](metadata=metadata, multiple_choice=mc)
        task_type = template_to_task_type[template]

        data = set()
        actual_name = model_name.lower()
        model_name = "llama" if "llama" in actual_name else "gemma" if "gemma" in actual_name else "qwen" if "qwen" in actual_name else "olmo" if "olmo" in actual_name else "gpt4omini" if "gpt" in actual_name else "unknown model"
        dataset_name = model_name + f"_{task_type}.jsonl"
        print(dataset_name)
        if mc:
            if not os.path.exists("./results_mc"):
                os.makedirs("./results_mc")
            directory = "./results_mc/" + dataset_name
        else:
            if not os.path.exists("./results"):
                os.makedirs("./results")
            directory = "./results/" + dataset_name
        if os.path.exists(directory):
            with open(directory, "r") as file:
                for line in file:
                    raw = json.loads(line)
                    data.add(raw["q"])

        counter = len(data)
        task_store = TaskStore(schema=generator.schema)
        generator.enumerate_task_plans(task_store)
        tasks = list(task_store)
        if num_tasks < len(tasks):
            tasks = random.sample(tasks, num_tasks)

        # counter = 0
        with open(directory, "a") as jsonl_file:
            for task in tqdm(tasks):
                # question, options, answer, _ = generator._generate_task(task)
                task = generator.generate(task)
                question, options, answer = task["question"], task["options"], task["answer"]
                if question in data:
                    # counter += 1
                    continue

                output = qa(model, question, options=options.values()
                            ) if mc else qa(model, question, options=None)
                if not mc:
                    # data_instance = {"q": question, "answer_only": output["answer_only"],
                    #                  "reasoning": output["reasoning"], "code": output["code"],
                    #                  "ground_truth": answer}
                    
                    data_instance = {"q": question, "answer_only": output["answer_only"],
                                     "reasoning": output["reasoning"],
                                     "ground_truth": answer}
                else:
                    selection = get_selection(output["answer"], options)
                    data_instance = {
                        "q": question, "raw_answer": output["answer"], "selection": selection}
                jsonl_file.write(json.dumps(data_instance))
                jsonl_file.write("\n")
                counter += 1
                if counter == num_tasks:
                    break
                # if counter % 5 == 0:
                    # print(f"Generated {counter} out of {num_tasks} tasks")
                # print(f"Free form answer for question {question}: ", output["free_form_answer"])


pattern = r"\{([^}]*)\}"

def get_selection(raw, options: dict, tolerance=0.001):
    raw_matches = list(re.finditer(pattern, raw))
    for match in raw_matches:
        extracted_val = match.group(1)
        # strip away all non-digits or decimal points from the extracted value
        extracted_val = re.sub(r'[^\d.]', '', extracted_val)
        try:
            extracted_val = round(float(extracted_val), 3)
            for option, val in options.items():
                if abs(round(float(extracted_val), 3) - round(float(val), 3)) <= tolerance:
                    return option
        except ValueError:
            print(
                f"This should never happen: ValueError - Couldn't parse the extracted value {extracted_val} into a float")
        
    return "unparsable"


def count_correct(raw_list, ground_truth, tolerance=0.001):
    correct_count = 0
    no_matches = 0
    unparsable = 0
    for raw, truth in zip(raw_list, ground_truth):
        matches = list(re.finditer(pattern, raw))
        if not matches:
            no_matches += 1
            continue
        for match in matches:
            extracted_val = match.group(1)
            # strip away all non-digits or decimal points from the extracted value
            extracted_val = re.sub(r'[^\d.]', '', extracted_val)
            parsable = False
            try:
                if abs(round(float(extracted_val), 3) - round(float(truth), 3)) <= tolerance:
                    correct_count += 1
                    break
                parsable = True
            except ValueError:
                pass
            
            if not parsable:
                unparsable = 1

    return correct_count, no_matches, unparsable


def llm_eval(raw_list, ground_truth):
    correct = 0
    # no_matches = 0
    for raw, gt in zip(raw_list, ground_truth):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You will help evaluate the accuracy of a model's answer."},
                {"role": "user", "content": f"Return 'yes' if the following string \"{raw}\" contains the ground_truth answer \"{gt}\", otherwise return no."},
                #  If you said 'yes', also return 'yes_match' if the answer was contained in the curly braces, otherwise return 'no_match'.
            ], max_tokens=4, temperature=0.9, top_p=1)

        if completion.choices[0].message.content == "yes":
            correct += 1
    return correct, 0


def eval_models(llm_based_eval=False, mc=False, dataset=None):
    output_file_name = "accuracies"
    if mc:
        output_file_name += "_mc_"
    else:
        output_file_name += "_open_answer_"

    if llm_based_eval:
        output_file_name += "llm_eval.jsonl"
    else:
        output_file_name += "heuristic_eval.jsonl"
        
    if not mc:
        results_dir = "results_open_answer/"
    else:
        results_dir = "results_mc/"

    if dataset is None:
        result_paths = [f for f in os.listdir(
            results_dir) if os.path.isfile(os.path.join(results_dir, f))]
    else:
        result_paths = [dataset]

    for path in result_paths:
        with open("./" + results_dir + path, "r") as jsonl_file:
            data = [json.loads(line) for line in jsonl_file]

        dataset_name = path.split("/")[-1].split(".")[0]
        total = len(data)

        if not mc:
            raw_answers_only = [d["answer_only"] for d in data]
            raw_reasoning_answers = [d["reasoning"] for d in data]
            raw_code_answers = [d["code"] for d in data]
            ground_truths = [d["ground_truth"] for d in data]

            if not llm_based_eval:
                correct_answer_only, no_matches_answer_only, unparsable_answer_only = count_correct(
                    raw_answers_only, ground_truths)
                correct_reasoning_answer, no_matches_reasoning, unparsable_reasoning = count_correct(
                    raw_reasoning_answers, ground_truths)
                correct_code_answer, no_matches_code, unparsable_code = count_correct(
                    raw_code_answers, ground_truths)
            else:
                correct_answer_only, no_matches_answer_only = llm_eval(
                    raw_answers_only, ground_truths)
                correct_reasoning_answer, no_matches_reasoning = llm_eval(
                    raw_reasoning_answers, ground_truths)
                correct_code_answer, no_matches_code = llm_eval(
                    raw_code_answers, ground_truths)

            answer_only_incorrect = total - correct_answer_only
            reasoning_incorrect = total - correct_reasoning_answer
            code_incorrect = total - correct_code_answer

            if answer_only_incorrect == 0:
                answer_only_incorrect = 1
            if reasoning_incorrect == 0:
                reasoning_incorrect = 1
            if code_incorrect == 0:
                code_incorrect = 1

            result = {"name": dataset_name, "answer_only": correct_answer_only / total,
                      "reasoning": correct_reasoning_answer / total, "code": correct_code_answer / total}
            if not llm_based_eval:
                result.update({"no_matches_answer_only": no_matches_answer_only / answer_only_incorrect,
                              "no_matches_reasoning": no_matches_reasoning / reasoning_incorrect, "no_matches_code": no_matches_code / code_incorrect,
                              "unparsable_answer_only": unparsable_answer_only / answer_only_incorrect, "unparsable_reasoning": unparsable_reasoning / reasoning_incorrect,
                              "unparsable_code": unparsable_code / code_incorrect})
        else:
            selections = [d["selection"] for d in data]
            correct, unparsable = 0, 0
            incorrect_options = {}
            for selection in selections:
                if selection == "unparsable":
                    unparsable += 1
                elif selection == "correct":
                    correct += 1
                else:
                    if selection in incorrect_options:
                        incorrect_options[selection] += 1
                    else:
                        incorrect_options[selection] = 1

            result = {"name": dataset_name, "correct": correct /
                      total, "unparsable": unparsable / total}
            
            for key in incorrect_options:
                result[key] = incorrect_options[key] / total

        for key in result:
            if key == "name":
                continue
            result[key] = round(result[key], 2)

        with open(output_file_name, "a") as f:
            f.write(json.dumps(result))
            f.write("\n")


# def visualize_results(accuracy_file):
#     with open(accuracy_file, "r") as f:
#         data = [json.loads(line) for line in f]


if __name__ == "__main__":
    generate_output(gemma.model, gemma.model_name, 25, "algebra", mc=False)
    # eval_models(llm_based_eval=False, mc=False)
