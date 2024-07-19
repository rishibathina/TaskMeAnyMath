import random
from geometry_task import MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator, CircleGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator, ArcLengthGenerator,TaskStore

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

    ArcLength_metadata = MathTemplateMetaData(template_path=ArcLength_template_path)

    ArcLength_task_store = TaskStore(schema = ArcLengthGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=ArcLength_template_path)
    metadata = MathTemplateMetaData(template_path=ArcLength_template_path)

    ArcLength_generator = ArcLengthGenerator(metadata=ArcLength_metadata)
    ArcLength_generator.enumerate_task_plans(ArcLength_task_store)
 
    # Select and print a random task
    ArcLength_tasks = list(ArcLength_task_store)
    if ArcLength_tasks:
        random_ArcLength_task = random.choice(ArcLength_tasks)
        ArcLength_task = ArcLength_generator.generate(random_ArcLength_task)
        print("Random ArcLength Task:")
        print("Question:", ArcLength_task['question'])
        print("Answer:", ArcLength_task['answer'])
if __name__ == "__main__":
    main()