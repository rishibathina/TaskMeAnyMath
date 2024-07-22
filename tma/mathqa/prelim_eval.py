import random
from geometry_task import MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator, CircleGenerator, AngleGenerator, TaskStore
from transformers import pipeline
import re

def main():
    pattern = r'\{([^}]*)\}'
    template_path = 'annotations/math_annotations/circle_templates.json'
    metadata = MathTemplateMetaData(template_path=template_path)
    generator = CircleGenerator(metadata=metadata)
    Circle_task_store = TaskStore(schema=CircleGenerator.schema)
    generator.enumerate_task_plans(Circle_task_store)
    Circle_tasks = list(Circle_task_store)
    task_instances = random.sample(Circle_tasks, 20)
    score = 0
    for task_instance in task_instances:
        task = generator.generate(task_instance)
        question = task['question']
        prompt = question + "\nPut your final answer in \{\}."
        model_name = "google/gemma-2-9b-it"
        # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        # model_name = "Qwen/Qwen2-7B-Instruct
        pipe = pipeline("text-generation", model=model_name, device_map="auto", max_new_tokens=100, return_full_text=False, temperature=0.7)
        messages = [{"role": "user", "content": prompt}]
        output = pipe(messages)[0]["generated_text"]
        print("Question:", question)
        print("Answer:", task['answer'])
        print("Output:", output)
        match = re.search(pattern, output)
        if match:
            if match.group(1) == task['answer']:
                # scores[model_name] += 1
                score += 1
    
    print(score)
 
if __name__ == "__main__":
    main()
    