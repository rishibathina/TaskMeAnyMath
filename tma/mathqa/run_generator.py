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

    PythagoreanTheorem_metadata = MathTemplateMetaData(template_path=PythagoreanTheorem_template_path)

    PythagoreanTheorem_task_store = TaskStore(schema = PythagoreanTheoremGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=PythagoreanTheorem_template_path)
    metadata = MathTemplateMetaData(template_path=PythagoreanTheorem_template_path)

    PythagoreanTheorem_generator = PythagoreanTheoremGenerator(metadata=PythagoreanTheorem_metadata)
    PythagoreanTheorem_generator.enumerate_task_plans(PythagoreanTheorem_task_store)
 
    # Select and print a random task
    PythagoreanTheorem_tasks = list(PythagoreanTheorem_task_store)
    if PythagoreanTheorem_tasks:
        random_PythagoreanTheorem_task = random.choice(PythagoreanTheorem_tasks)
        PythagoreanTheorem_task = PythagoreanTheorem_generator.generate(random_PythagoreanTheorem_task)
        print("Random PythagoreanTheorem Task:")
        print("Question:", PythagoreanTheorem_task['question'])
        print("Answer:", PythagoreanTheorem_task['answer'])
if __name__ == "__main__":
    main()