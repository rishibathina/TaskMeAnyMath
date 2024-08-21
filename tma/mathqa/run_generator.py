import random
from geometry_task import (
    MathTemplateMetaData, PerimeterGenerator, MidpointGenerator,IntersectionGenerator, 
    PerimeterGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator,
     ArcLengthGenerator, CircleGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator,
    PointSlopeGenerator, RemainderTheoremGenerator, QuadraticFormulaGenerator, TaskStore
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
    PointSlope_template_path = r"C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\point_slope_templates.json"
    RemainderTheorem_template_path = r"C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\remainderTheorem_template.json"
    QuadraticFormula_template_path = r"C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\quadraticFormula_templates.json"

    QuadraticFormula_metadata = MathTemplateMetaData(template_path=QuadraticFormula_template_path)

    QuadraticFormula_task_store = TaskStore(schema = QuadraticFormulaGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=QuadraticFormula_template_path)
    metadata = MathTemplateMetaData(template_path=QuadraticFormula_template_path)

    QuadraticFormula_generator = QuadraticFormulaGenerator(metadata=QuadraticFormula_metadata)
    QuadraticFormula_generator.enumerate_task_plans(QuadraticFormula_task_store)
 
    # Select and print a random task
    QuadraticFormula_tasks = list(QuadraticFormula_task_store)
    if QuadraticFormula_tasks:
        random_QuadraticFormula_task = random.choice(QuadraticFormula_tasks)
        QuadraticFormula_task = QuadraticFormula_generator.generate(random_QuadraticFormula_task)
        print("Random QuadraticFormula Task:")
        print("Question:", QuadraticFormula_task['question'])
        print("Options:", QuadraticFormula_task['options'])
        print("Answer:", QuadraticFormula_task['answer'])
if __name__ == "__main__":
    main()