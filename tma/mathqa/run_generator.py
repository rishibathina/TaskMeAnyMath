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

    Intersection_metadata = MathTemplateMetaData(template_path=Intersection_template_path)

    Intersection_task_store = TaskStore(schema = IntersectionGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=Intersection_template_path)
    metadata = MathTemplateMetaData(template_path=Intersection_template_path)

    Intersection_generator = IntersectionGenerator(metadata=Intersection_metadata)
    Intersection_generator.enumerate_task_plans(Intersection_task_store)
 
    # Select and print a random task
    Intersection_tasks = list(Intersection_task_store)
    if Intersection_tasks:
        random_Intersection_task = random.choice(Intersection_tasks)
        Intersection_task = Intersection_generator.generate(random_Intersection_task)
        print("Random Intersection Task:")
        print("Question:", Intersection_task['question'])
        print("Options:", Intersection_task['options'])
        print("Answer:", Intersection_task['answer'])
if __name__ == "__main__":
    main()