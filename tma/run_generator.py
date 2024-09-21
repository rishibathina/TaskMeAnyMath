import random
from geometry_task import AngleGenerator, AngleSumGenerator, ArcLengthGenerator, CircleGenerator, ConeVolumeGenerator, IntersectionGenerator, MathTemplateMetaData, MidpointGenerator, PerimeterGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator, SideLengthGenerator, TaskStore, TriangleAreaGenerator, VolumeRectangularPrismGenerator, VolumeSphereGenerator
from algebra_task import PointSlopeGenerator, QuadraticFormulaGenerator, RemainderTheoremGenerator, ExponentialDecayGenerator, PolynomialFactoringGenerator
from stats_task import MeanMedianModeGenerator, BayesTheoremGenerator, SampleStdDeviationGenerator, PopulationStdDeviationGenerator, CrossEntropyGenerator
import os


def main():
    templates = ["cross_entropy_templates.json"]
    mc = True
    for template in templates:
        template_path = "./math_annotations/" + template
        metadata = MathTemplateMetaData(template_path=template_path)
        if "circle" in template:
            generator = CircleGenerator(metadata=metadata, multiple_choice=mc)
        elif "angle_sum" in template:
            generator = AngleSumGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "angle_tem" in template:
            generator = AngleGenerator(metadata=metadata, multiple_choice=mc)
        elif "midpoint" in template:
            generator = MidpointGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "intersection" in template:
            generator = IntersectionGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "perimeter" in template:
            generator = PerimeterGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "area" in template:
            generator = TriangleAreaGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "sideLength" in template:
            generator = SideLengthGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "cone" in template:
            generator = ConeVolumeGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "sphere" in template:
            generator = VolumeSphereGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "point_distance" in template:
            generator = PointDistanceGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "pythagorean" in template:
            generator = PythagoreanTheoremGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "rectangular" in template:
            generator = VolumeRectangularPrismGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "arcLength" in template:
            generator = ArcLengthGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "slope" in template:
            generator = PointSlopeGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "remainder" in template:
            generator = RemainderTheoremGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "quadratic" in template:
            generator = QuadraticFormulaGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "exponential" in template:
            generator = ExponentialDecayGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "polynomial" in template:
            generator = PolynomialFactoringGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "mode" in template:
            generator = MeanMedianModeGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "bayes" in template:
            generator = BayesTheoremGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "sample" in template:
            generator = SampleStdDeviationGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "population" in template:
            generator = PopulationStdDeviationGenerator(
                metadata=metadata, multiple_choice=mc)
        elif "cross" in template:
            generator = CrossEntropyGenerator(
                metadata=metadata, multiple_choice=mc)

    
        task_store = TaskStore(schema=generator.schema)
        generator.enumerate_task_plans(task_store)
        tasks = list(task_store)
        if tasks:
            task = random.choice(tasks)
            task = generator.generate(task)
            print("Question:", task["question"])
            print("Options:", task["options"])
            print("Answer:", task["answer"])


if __name__ == "__main__":
    main()
