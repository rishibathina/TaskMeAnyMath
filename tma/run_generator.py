import random
from geometry_task import MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator, CircleGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator, ArcLengthGenerator,TaskStore

def main():

    # Define the path to templates using a raw string
    template_path = 'annotations/math_annotations/circle_templates.json'

    metadata = MathTemplateMetaData(template_path=template_path)
    generator = CircleGenerator(metadata=metadata)
    task_store = TaskStore(schema=CircleGenerator.schema)
    generator.enumerate_task_plans(task_store)
 
    Circle_tasks = list(task_store)
    task_instances = random.sample(Circle_tasks, 10)
    for task_instance in task_instances:
        task = generator.generate(task_instance)
        question = task['question']
        print(question + "\nPut your final answer in \{\}.")
        print("Answer:", task['answer'])
        
if __name__ == "__main__":
    main()
    