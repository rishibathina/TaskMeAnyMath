import random
from metadata import MathTemplateMetaData
from task_store import TaskStore
from constant import template_to_generator, template_to_task_type
import os

def main():
    mc = True
    math_field = "algebra"
    # math_field = "geometry"
    if math_field == "geometry":
        templates = [f for f in os.listdir("../math_annotations/geometry") if os.path.isfile(os.path.join("../math_annotations/geometry", f))]
        print(templates)
        templates.remove("cone_volume_templates.json")
        templates.remove("perpendicular_template.json")
        templates.remove("angle_templates.json")
        
        templates.remove("midpoint_templates.json")
        templates.remove("intersection_templates.json")
        
        templates.remove("cone_volume_templates.json")

    elif math_field == "algebra":
        templates = [f for f in os.listdir("../math_annotations/algebra") if os.path.isfile(os.path.join("../math_annotations/algebra", f))]
        print(templates)
    else:
        raise ValueError("Invalid math field")
     
    for template in templates:
        template_path = "../math_annotations/" + math_field + "/" + template
        metadata = MathTemplateMetaData(template_path=template_path)
        generator = template_to_generator[template](metadata=metadata, multiple_choice=mc)
        
        task_store = TaskStore(schema=generator.schema)
        generator.enumerate_task_plans(task_store)
        tasks = list(task_store)
        if tasks:
            task_instances = random.choices(tasks, k=2)
            for task_instance in task_instances: 
                task = generator.generate(task_instance)
                print("Question:", task["question"])
                print("Options:", task["options"])
                print("Answer:", task["answer"])
        print("\n")

if __name__ == "__main__":
    main()
