import random
from geometry_task import MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator, PerimeterGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator, ArcLengthGenerator, CircleGenerator, TaskStore

def main():
    # Define the path to templates using a raw string
    Circle_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\circle_templates.json'
    Angle_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\angle_template.json'
    Perimeter_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\perimeter_templates.json'
    Midpoint_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\midpoint_templates.json'
    Intersection_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\intersection_templates.json'
    Perpendicular_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\perpendicular_template.json'
    SideLength_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\sideLength_template.json'
    ArcLength_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\arcLength_template.json'
    Circle_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\circle_templates.json'

    Perimeter_metadata = MathTemplateMetaData(template_path=Perimeter_template_path)

    Perimeter_task_store = TaskStore(schema = PerimeterGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=Perimeter_template_path)
    metadata = MathTemplateMetaData(template_path=Perimeter_template_path)

    Perimeter_generator = PerimeterGenerator(metadata=Perimeter_metadata)
    Perimeter_generator.enumerate_task_plans(Perimeter_task_store)
 
    # Select and print a random task
    Perimeter_tasks = list(Perimeter_task_store)
    if Perimeter_tasks:
        random_Perimeter_task = random.choice(Perimeter_tasks)
        Perimeter_task = Perimeter_generator.generate(random_Perimeter_task)
        print("Random Perimeter Task:")
        print("Question:", Perimeter_task['question'])
        print("Answer:", Perimeter_task['answer'])
if __name__ == "__main__":
    main()