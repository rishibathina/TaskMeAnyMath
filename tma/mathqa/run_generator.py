import random
from geometry_task import (
    MathTemplateMetaData, PerimeterGenerator, MidpointGenerator,IntersectionGenerator, 
    PerimeterGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator,
     ArcLengthGenerator, CircleGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator, TaskStore
)

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
    PointDistance_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\point_distance_template.json'
    PythagoreanTheorem_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\pythagoreanTheorem_template.json'

    Midpoint_metadata = MathTemplateMetaData(template_path=Midpoint_template_path)

    Midpoint_task_store = TaskStore(schema = MidpointGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=Midpoint_template_path)
    metadata = MathTemplateMetaData(template_path=Midpoint_template_path)

    Midpoint_generator = MidpointGenerator(metadata=Midpoint_metadata)
    Midpoint_generator.enumerate_task_plans(Midpoint_task_store)
 
    # Select and print a random task
    Midpoint_tasks = list(Midpoint_task_store)
    if Midpoint_tasks:
        random_Midpoint_task = random.choice(Midpoint_tasks)
        Midpoint_task = Midpoint_generator.generate(random_Midpoint_task)
        print("Random Midpoint Task:")
        print("Question:", Midpoint_task['question'])
        print("Options:", Midpoint_task['options'])
        print("Answer:", Midpoint_task['answer'])
if __name__ == "__main__":
    main()