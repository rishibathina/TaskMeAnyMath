from typing import Dict, List, Tuple
import json
import numpy as np

from base import TaskGenerator
from task_store import TaskStore
from utils import *
from metadata import MathTemplateMetaData


class GeoPlanGenerator(TaskGenerator):
    metadata: MathTemplateMetaData

    def __init__(self, metadata: MathTemplateMetaData, seed=42):
        super().__init__(metadata, seed=seed)

    def _task_plan_to_str(self, task_plan) -> str:
        "(Abstract method) task plan to string"

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        "(Abstract method) generate task"

    def generate(self, task_plan, return_data=True, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        question, options, answer, math_metadata = self._generate_task(
            task_plan)

        task = {
            "question": question.replace("_", " "),
            "options": options if self.multiple_choice else None,
            "answer": answer.replace("_", " "),
            "task_plan": self._task_plan_to_str(task_plan),
            "math_metadata": math_metadata,
        }
        return task


class TrianglePerimeterGenerator(GeoPlanGenerator):
    schema = {
        "question_template": "str",
        "side_one": "str",
        "side_two": "str",
        "side_three": "str",
    }

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=False,
        seed=42,
        side_one_range=PERI_SIDE_ONE_RANGE,
        side_two_range=PERI_SIDE_TWO_RANGE,
        side_three_range=PERI_SIDE_THREE_RANGE,
        num_splices=5,
    ):
        super().__init__(metadata, seed=seed)
        self.side_one_range = side_one_range
        self.side_two_range = side_two_range
        self.side_three_range = side_three_range
        self.int_to_peri_list = int_to_peri_list
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        single = make_single_prod(self.side_one_range, self.num_splices)
        pairs = make_pair_prod(
            self.side_one_range, self.side_three_range, self.num_splices
        )
        triplets = make_triplet_prod(
            self.side_one_range,
            self.side_two_range,
            self.side_three_range,
            self.num_splices,
        )

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            peri_list = locals()[self.int_to_peri_list[param_count]]

            for template in templates:
                for group in peri_list:
                    if isinstance(group, (float, np.float64)):
                        group = [group]
                    params = [
                        str(param) if param is not None else None for param in group
                    ]
                    while len(params) < 3:
                        params.append(None)

                    task_plan = {
                        "question_template": template,
                        "side_one": params[0],
                        "side_two": params[1],
                        "side_three": params[2],
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        side_one = round(float(task_plan["side_one"]), 2)
        side_two = (
            round(float(task_plan["side_two"]), 2)
            if task_plan["side_two"] is not None
            else None
        )
        side_three = (
            round(float(task_plan["side_three"]), 2)
            if task_plan["side_three"] is not None
            else None
        )

        if side_two is None:
            question = template.format(side_one=side_one)
            answer = str(round(3 * side_one, 2))
            if self.multiple_choice:
                option2 = str(round(2 * side_one, 2))
                option3 = str(round(side_one, 2))
                option4 = str(round(side_one * side_one / 2, 2))
                all_options = [option2, option3, option4] 
        elif side_three is None:
            question = template.format(side_one=side_one, side_two=side_two)
            answer = str(round(2 * side_one + side_two, 2))
            if self.multiple_choice:
                option2 = str(round(3 * side_one, 2))
                option3 = str(round(side_one + 2 * side_two, 2))
                option4 = str(round(side_one * side_two / 2, 2))
                all_options = [option2, option3, option4]
        else:
            question = template.format(side_one=side_one, side_two=side_two, side_three=side_three)
            answer = str(round(side_one + side_two + side_three, 2))
            if self.multiple_choice:
                option2 = str(round(3 * side_one, 2))
                option3 = str(round(side_one + 2 * side_two, 2))
                option4 = str(round(side_one * side_two / 2, 2))
                all_options = [option2, option3, option4]
        
        if self.multiple_choice:          
            deduped_options = [opt for opt in dict.fromkeys(all_options) if opt != answer]
            deduped_options_dict = {f"incorrect option {i+1}": opt for i, opt in enumerate(dict.fromkeys(deduped_options))}
            deduped_options_dict["correct"] = answer
        else:
            deduped_options_dict = {}
            
        return question, deduped_options_dict, answer, self.metadata


class MidpointGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 100)
    Y_COORDINATE_RANGE = (0, 100)
    Z_COORDINATE_RANGE = (0, 100)

    schema = {"question_template": "str", "point1": "str", "point2": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        x_range=X_COORDINATE_RANGE,
        y_range=Y_COORDINATE_RANGE,
        z_range=Z_COORDINATE_RANGE,
        num_splices=5,
    ):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(
            self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(
            self.y_range[0], self.y_range[1], self.num_splices)
        z_values = np.linspace(
            self.z_range[0], self.z_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for x1 in x_values:
                    for y1 in y_values:
                        for z1 in z_values:
                            for x2 in x_values:
                                for y2 in y_values:
                                    for z2 in z_values:
                                        task_plan = {
                                            "question_template": template,
                                            "point1": (x1, y1, z1),
                                            "point2": (x2, y2, z2),
                                        }
                                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        point1 = eval(task_plan["point1"])
        point2 = eval(task_plan["point2"])
        midpoint = (
            round((point1[0] + point2[0]) / 2, 2),
            round((point1[1] + point2[1]) / 2, 2),
            round((point1[2] + point2[2]) / 2, 2),
        )
        question = template.format(param1=point1, param2=point2)
        correct_answer = "({}, {}, {})".format(midpoint[0], midpoint[1], midpoint[2])
        if self.multiple_choice:
            option1 = "({}, {}, {})".format(
                round(point1[0] + point2[0], 2),
                round(point1[1] + point2[1], 2),
                round(point1[2] + point2[2], 2),
            )
            option2 = "({}, {}, {})".format(
                round(midpoint[0], 2),
                round((point1[1] + point2[1]) / 3, 2),
                round(midpoint[2], 2),
            )
            option3 = "({}, {}, {})".format(
                round((point1[0] + point2[0]) / 2, 2),
                round(midpoint[2], 2),
                round(midpoint[1], 2),
            )
            all_options = [option1, option2, option3]
            deduped_options = {f"incorrect option {i+1}": opt for i, opt in enumerate(dict.fromkeys(all_options))}
            deduped_options["correct"] = correct_answer
        else:
            deduped_options = {}
        return question, deduped_options, correct_answer, self.metadata


class IntersectionGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 10)
    Y_COORDINATE_RANGE = (0, 10)

    schema = {
        "question_template": "str",
        "vector1_start": "str",
        "vector1_end": "str",
        "vector2_start": "str",
        "vector2_end": "str",
    }

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        x_range=X_COORDINATE_RANGE,
        y_range=Y_COORDINATE_RANGE,
        num_splices=4,
    ):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(
            self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(
            self.y_range[0], self.y_range[1], self.num_splices)
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                for x3 in x_values:
                                    for y3 in y_values:
                                        for x4 in x_values:
                                            for y4 in y_values:
                                                task_plan = {
                                                    "question_template": template,
                                                    "vector1_start": str((x1, y1)),
                                                    "vector1_end": str((x2, y2)),
                                                    "vector2_start": str((x3, y3)),
                                                    "vector2_end": str((x4, y4)),
                                                }
                                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        vector1_start = eval(task_plan["vector1_start"])
        vector1_end = eval(task_plan["vector1_end"])
        vector2_start = eval(task_plan["vector2_start"])
        vector2_end = eval(task_plan["vector2_end"])
        vector1 = np.array(vector1_end) - np.array(vector1_start)
        vector2 = np.array(vector2_end) - np.array(vector2_start)

        if np.cross(vector1, vector2) == 0:
            answer = "null"
            deduped_options = {}
        else:
            A = np.array([[vector1[0], -vector2[0]],
                        [vector1[1], -vector2[1]]])
            b = np.array([
                vector2_start[0] - vector1_start[0],
                vector2_start[1] - vector1_start[1],
            ])
            t = np.linalg.solve(A, b)
            intersection = vector1_start + t[0] * vector1
            answer = str(tuple(round(coord, 2) for coord in intersection))

            if self.multiple_choice:
                option1 = str(tuple(round(coord, 2) for coord in vector1_start))
                option2 = str(tuple(round(coord, 2) for coord in vector1_end))
                option3 = str(tuple(round(coord, 2) for coord in vector2_start))
                option4 = str(tuple(round(coord, 2) for coord in vector2_end))
                all_options = [option1, option2, option3, option4]
                deduped_options = {f"incorrect option {i+1}": opt for i, opt in enumerate(dict.fromkeys(all_options))}
                deduped_options["correct"] = answer
            else:
                deduped_options = {}

        question = template.format(
            param1=vector1_start,
            param2=vector1_end,
            param3=vector2_start,
            param4=vector2_end,
        )
        return question, deduped_options, answer, self.metadata


class PerpendicularGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 10)
    Y_COORDINATE_RANGE = (0, 10)

    schema = {
        "question_template": "str",
        "vector1_start": "str",
        "vector1_end": "str",
        "vector2_start": "str",
        "vector2_end": "str",
    }

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        seed=42,
        x_range=X_COORDINATE_RANGE,
        y_range=Y_COORDINATE_RANGE,
        num_splices=4,
    ):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(
            self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(
            self.y_range[0], self.y_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                for x3 in x_values:
                                    for y3 in y_values:
                                        for x4 in x_values:
                                            for y4 in y_values:
                                                task_plan = {
                                                    "question_template": template,
                                                    "vector1_start": str((x1, y1)),
                                                    "vector1_end": str((x2, y2)),
                                                    "vector2_start": str((x3, y3)),
                                                    "vector2_end": str((x4, y4)),
                                                }
                                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        template = task_plan["question_template"]
        vector1_start = eval(task_plan["vector1_start"])
        vector1_end = eval(task_plan["vector1_end"])
        vector2_start = eval(task_plan["vector2_start"])
        vector2_end = eval(task_plan["vector2_end"])
        vector1 = np.array(vector1_end) - np.array(vector1_start)
        vector2 = np.array(vector2_end) - np.array(vector2_start)

        dot_product = np.dot(vector1, vector2)
        if dot_product == 0:
            answer = "Yes,"
        else:
            answer = "No"

        question = template.format(
            param1=vector1_start,
            param2=vector1_end,
            param3=vector2_start,
            param4=vector2_end,
        )
        return question, answer, self.metadata


class SquareSideLengthGenerator(GeoPlanGenerator):
    AREA_RANGE = (1, 1000)

    schema = {"question_template": "str", "area": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        area_range=AREA_RANGE,
        num_splices=50,
    ):
        super().__init__(metadata, seed=seed)
        self.area_range = area_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        area_values = np.linspace(
            self.area_range[0], self.area_range[1], self.num_splices
        )

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for area in area_values:
                    task_plan = {
                        "question_template": template, "area": str(area)}
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        area = round(float(task_plan["area"]), 2)
        side_length = round(np.sqrt(area), 2)
        question = template.format(area=area)
        answer = str(side_length)
        if self.multiple_choice:
            option1 = str(round(area / 2, 2))
            option2 = str(round(side_length * 2, 2))
            option3 = str(round(np.sqrt(area / 2), 2))
            all_options = [option1, option2, option3]
            deduped_options = {f"incorrect option {i+1}": opt for i, opt in enumerate(dict.fromkeys(all_options))}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class ArcLengthGenerator(GeoPlanGenerator):
    RADIUS_RANGE = (1, 30)
    ANGLE_RANGE = (0, 360)

    schema = {"question_template": "str", "radius": "str", "angle": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        radius_range=RADIUS_RANGE,
        angle_range=ANGLE_RANGE,
        num_splices=10,
    ):
        super().__init__(metadata, seed=seed)
        self.radius_range = radius_range
        self.angle_range = angle_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        radius_values = np.linspace(
            self.radius_range[0], self.radius_range[1], self.num_splices
        )
        angle_values = np.linspace(
            self.angle_range[0], self.angle_range[1], self.num_splices
        )

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for radius in radius_values:
                    for angle in angle_values:
                        task_plan = {
                            "question_template": template,
                            "radius": str(radius),
                            "angle": str(angle),
                        }
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        radius = round(float(task_plan["radius"]), 2)
        angle = round(float(task_plan["angle"]), 2)
        arc_length = round((np.pi * radius * angle) / 180, 2)
        question = template.format(radius=radius, angle=angle)
        answer = str(arc_length)

        if self.multiple_choice:
            option2 = str(round(np.pi * radius * angle / 360, 2))
            option3 = str(round(np.pi * radius**2 * angle / 180, 2))
            option4 = str(round(np.pi * angle / 180, 2))
            all_options = [option2, option3, option4]
            deduped_options = {f"incorrect option {i+1}": opt for i, opt in enumerate(dict.fromkeys(all_options))}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class CircleGenerator(GeoPlanGenerator):
    schema = {
        "question_template": "str",
        "radius": "str",
        "circumference": "str",
        "area": "str",
    }

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        seed=42,
        num_tasks=100,
        multiple_choice=False,
    ):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store: TaskStore):
        all_values = np.linspace(
            1, 100, self.num_tasks
        )
        template_breakdown = self.metadata.templates_by_num_params
        for _, texts in template_breakdown.items():
            for text in texts:
                for value in all_values:
                    if "circumference" in text or "perimeter" in text:
                        radius = value / (2 * np.pi)
                        circumference = value
                        area = np.pi * radius**2
                    elif "area" in text:
                        radius = np.sqrt(value / np.pi)
                        circumference = 2 * np.pi * radius
                        area = value
                    else:  # default to radius if not specified
                        radius = value
                        circumference = 2 * np.pi * radius
                        area = np.pi * radius**2

                    result = (radius, circumference, area)
                    if result not in self.all_tasks:
                        task_plan = {
                            "question_template": text,
                            "radius": str(radius),
                            "circumference": str(circumference),
                            "area": str(area),
                        }
                        task_store.add(task_plan)
                        self.all_tasks.add(result)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        radius = round(float(task_plan["radius"]), 2)
        circumference = round(float(task_plan["circumference"]), 2)
        area = round(float(task_plan["area"]), 2)
        options = {}

        if "circumference" in template or "perimeter" in template:
            if "radius" in template:
                question = template.format(param1=radius)
                answer = str(round(circumference, 2))
                if self.multiple_choice:
                    option1 = str(round(np.pi * radius, 2))  # Forgetting the 2
                    option2 = str(round(0.5 * np.pi * radius, 2))
                    option3 = str(round(np.pi * radius**2, 2))  # Mistaking it for area
                    options = {f"incorrect option {i+1}": opt for i, opt in enumerate([option1, option2, option3])}
                    options["correct"] = answer
            else:
                question = template.format(param1=circumference)
                answer = str(round(circumference / (2 * np.pi), 2))
                if self.multiple_choice:
                    option1 = str(round(circumference / np.pi, 2))
                    option2 = str(round(circumference / (0.5 * np.pi), 2))
                    option3 = str(round(np.sqrt(circumference / np.pi), 2))
                    options = {f"incorrect option {i+1}": opt for i, opt in enumerate([option1, option2, option3])}
                    options["correct"] = answer
        elif "area" in template:
            if "radius" in template:
                question = template.format(param1=radius)
                answer = str(round(np.pi * radius**2, 2))
                if self.multiple_choice:
                    option1 = str(round(np.pi * radius, 2))
                    option2 = str(round(np.pi * radius**3, 2))
                    option3 = str(round(2 * np.pi * radius, 2))
                    options = {f"incorrect option {i+1}": opt for i, opt in enumerate([option1, option2, option3])}
                    options["correct"] = answer
            else:
                question = template.format(param1=area)
                answer = str(round(np.sqrt(area / np.pi), 2))
                if self.multiple_choice:
                    option1 = str(round(area / np.pi, 2))
                    option2 = str(round(2 * np.sqrt(area / np.pi), 2))
                    option3 = str(round(area / (2 * np.pi), 2))
                    options = {f"incorrect option {i+1}": opt for i, opt in enumerate([option1, option2, option3])}
                    options["correct"] = answer
        else:
            raise ValueError(
                "Template must specify either perimeter, area, or radius.")

        return question, options, answer, self.metadata


class TriangleAreaGenerator(GeoPlanGenerator):
    schema = {"question_template": "str", "base": "str", "height": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        seed=42,
        base_range=(1, 100),
        height_range=(1, 100),
        num_tasks=100,
        multiple_choice=False,
    ):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.base_range = base_range
        self.height_range = height_range
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store: TaskStore):
        template_breakdown = self.metadata.templates_by_num_params
        all_bases = np.linspace(
            self.base_range[0], self.base_range[1], self.num_tasks)
        all_heights = np.linspace(
            self.height_range[0], self.height_range[1], self.num_tasks
        )
        for _, templates in template_breakdown.items():
            for template_text in templates:
                for base in all_bases:
                    for height in all_heights:
                        task_plan = {
                            "question_template": template_text,
                            "base": str(base),
                            "height": str(height),
                        }
                        task_store.add(task_plan)
                        self.all_tasks.add((base, height))

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        question = task_plan["question_template"]
        base = round(float(task_plan["base"]), 2)
        height = round(float(task_plan["height"]), 2)
        question_text = question.format(base, height)
        area = str(round(0.5 * base * height, 2))
        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = str(round(base * height, 2))
            options["incorrect option 2"] = str(round(0.25 * base * height, 2))
            options["incorrect option 3"] = str(round(base * base, 2))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = area
        else:
            deduped_options = {}
        return question_text, deduped_options, area, self.metadata


class AngleSumGenerator(GeoPlanGenerator):
    schema = {"question_template": "str", "shape": "str", "sides": "int"}

    def __init__(self, metadata, seed=42, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.all_tasks = set()
        self.shapes = {
            "triangle": 3,
            "quadrilateral": 4,
            "pentagon": 5,
            "hexagon": 6,
            "heptagon": 7,
            "octagon": 8,
            "nonagon": 9,
            "decagon": 10,
        }
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        templates_by_num_params = self.metadata.templates_by_num_params
        for _, templates in templates_by_num_params.items():
            for template_text in templates:
                for shape, sides in self.shapes.items():
                    task_plan = {
                        "question_template": template_text,
                        "shape": shape,
                        "sides": sides,
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        shape = task_plan["shape"]
        sides = task_plan["sides"]
        problem = (shape, sides)
        self.all_tasks.add(problem)
        angle_sum = (sides - 2) * 180
        question = template.format(shape)
        answer = str(angle_sum)
        options = {}
        
        if self.multiple_choice:
            while len(options) < 3:
                wrong_shape = np.random.choice(list(self.shapes.keys()))
                wrong_sides = self.shapes[wrong_shape]
                wrong_angle_sum = (wrong_sides - 2) * 180
                if str(wrong_angle_sum) != answer and str(wrong_angle_sum) not in options.values():
                    options[f"incorrect option {len(options)+1}"] = str(wrong_angle_sum)
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        
        return question, deduped_options, answer, self.metadata


class RectangularPrismVolumeGenerator(GeoPlanGenerator):
    schema = {
        "question_template": "str",
        "length": "str",
        "width": "str",
        "height": "str",
    }

    def __init__(
        self,
        metadata,
        seed=42,
        length_range=(1, 10),
        width_range=(1, 10),
        height_range=(1, 10),
        num_tasks=100,
        multiple_choice=False,
    ):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.length_range = length_range
        self.width_range = width_range
        self.height_range = height_range
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        lengths = np.random.uniform(
            self.length_range[0], self.length_range[1], self.num_tasks
        )
        widths = np.random.uniform(
            self.width_range[0], self.width_range[1], self.num_tasks
        )
        heights = np.random.uniform(
            self.height_range[0], self.height_range[1], self.num_tasks
        )
        templates_by_num_params = self.metadata.templates_by_num_params
        for _, templates in templates_by_num_params.items():
            for template_text in templates:
                for length in lengths:
                    for width in widths:
                        for height in heights:
                            task_plan = {
                                "question_template": template_text,
                                "length": length,
                                "width": width,
                                "height": height,
                            }
                            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        height = round(float(task_plan["height"]), 2)
        width = round(float(task_plan["width"]), 2)
        length = round(float(task_plan["length"]), 2)
        problem = (height, width, length)
        self.all_tasks.add(problem)
        volume = round(length * width * height, 2)
        answer = str(volume)
        options = {}
        
        if self.multiple_choice:
            options["incorrect option 1"] = str(round(length * width * 2, 2))
            options["incorrect option 2"] = str(round(length**3, 2))
            options["incorrect option 3"] = str(round(length + width + height, 2))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        question = template.format(length, width, height)
        return question, deduped_options, answer, self.metadata


class AngleGenerator(GeoPlanGenerator):
    schema = {"question_template": "str",
              "angles": "list"}

    def __init__(self, metadata: MathTemplateMetaData, seed=42, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store: TaskStore):
        template_breakdown = self.metadata.templates_by_num_params
        shapes = {
            "equilateral triangle": 60,
            "rectangle": 90,
            "regular pentagon": 108,
            "regular hexagon": 120,
        }
        for _, templates in template_breakdown.items():
            for template_text in templates:
                for _, angles in shapes.items():
                    task_plan = {
                        "question_template": template_text, "angles": angles}
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        angles = task_plan["angles"]
        question = template.format(angles)
        
        if angles == 60:
            answer = "equilateral triangle"
        elif angles == 90:
            answer = "rectangle"
        elif angles == 108:
            answer = "regular pentagon"
        elif angles == 120:
            answer = "regular hexagon"
        else:
            answer = "unreasonable shape"

        options = {}
        if self.multiple_choice and answer != "unreasonable shape":
            options["incorrect option 1"] = "equilateral triangle" if answer != "equilateral triangle" else "rectangle"
            options["incorrect option 2"] = "rectangle" if answer != "rectangle" else "regular pentagon"
            options["incorrect option 3"] = "regular pentagon" if answer != "regular pentagon" else "regular hexagon"
            options["incorrect option 4"] = "regular hexagon" if answer != "regular hexagon" else "equilateral triangle"
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class SphereVolumeGenerator(GeoPlanGenerator):
    schema = {"question_template": "str", "radius": "str"}

    def __init__(
        self,
        metadata,
        seed=42,
        radius_range=(1, 20),
        num_tasks=100,
        multiple_choice=False,
    ):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.radius_range = radius_range
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        radii = np.random.uniform(
            self.radius_range[0], self.radius_range[1], self.num_tasks
        )
        templates_by_num_params = self.metadata.templates_by_num_params
        for _, templates in templates_by_num_params.items():
            for template_text in templates:
                for radius in radii:
                    task_plan = {
                        "question_template": template_text, "radius": radius}
                    if self._task_plan_to_str(task_plan) not in self.all_tasks:
                        self.all_tasks.add(self._task_plan_to_str(task_plan))
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        radius = round(float(task_plan["radius"]), 2)
        volume = round((4 / 3) * np.pi * (radius**3), 2)
        question = template.format(radius)
        answer = str(volume)
        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = str(round(np.pi * (radius**3), 2))
            options["incorrect option 2"] = str(round((1 / 3) * np.pi * (radius**3), 2))
            options["incorrect option 3"] = str(round((4 / 3) * np.pi * (radius**2), 2))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata


class ConeVolumeGenerator:
    schema = {"question_template": "str", "radius": "str", "height": "str"}

    def __init__(
        self,
        metadata,
        seed=42,
        radius_range=(1, 20),
        height_range=(1, 20),
        num_tasks=100,
        multiple_choice=False,
    ):
        super().__init__(metadata, seed=seed)
        self.metadata = metadata
        self.seed = seed
        self.radius_range = radius_range
        self.height_range = height_range
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan, sort_keys=True)

    def enumerate_task_plans(self, task_store):
        radii = np.random.uniform(
            self.radius_range[0], self.radius_range[1], self.num_tasks
        )
        heights = np.random.uniform(
            self.height_range[0], self.height_range[1], self.num_tasks
        )
        templates_by_num_params = self.metadata.templates_by_num_params
        for _, templates in templates_by_num_params.items():
            for template_text in templates:
                for radius in radii:
                    for height in heights:
                        task_plan = {
                            "question_template": template_text,
                            "radius": radius,
                            "height": height,
                        }
                        task_plan_str = self._task_plan_to_str(task_plan)
                        if task_plan_str not in self.all_tasks:
                            self.all_tasks.add(task_plan_str)
                            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        radius = round(float(task_plan["radius"]), 2)
        height = round(float(task_plan["height"]), 2)
        volume = round((1 / 3) * np.pi * (radius**2) * height, 2)
        question = template.format(radius, height)
        answer = str(volume)
        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = str(round((4 / 3) * np.pi * (radius**3), 2))
            options["incorrect option 2"] = str(round((4 / 3) * np.pi * (radius**2) * height, 2))
            options["incorrect option 3"] = str(round(np.pi * (radius**2) * height, 2))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata


class PointDistanceGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (-10, 10)
    Y_COORDINATE_RANGE = (-10, 10)
    schema = {"question_template": "str", "point1": "str", "point2": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        x_range=X_COORDINATE_RANGE,
        y_range=Y_COORDINATE_RANGE,
        num_splices=10,
    ):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(
            self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(
            self.y_range[0], self.y_range[1], self.num_splices)
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                if (x1, y1) != (x2, y2):
                                    task_plan = {
                                        "question_template": template,
                                        "point1": str((x1, y1)),
                                        "point2": str((x2, y2)),
                                    }
                                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        point1 = eval(task_plan["point1"])
        point2 = eval(task_plan["point2"])
        point1 = (round(point1[0], 2), round(point1[1], 2))
        point2 = (round(point2[0], 2), round(point2[1], 2))
        distance = round(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2), 2)
        question = template.format(point1=point1, point2=point2)
        answer = str(distance)
        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = str(round(abs(point2[0] - point1[0]) + abs(point2[1] - point1[1]), 2))
            options["incorrect option 2"] = str(round(abs(point2[0] - point1[0]), 2))
            options["incorrect option 3"] = str(round(abs(point2[1] - point1[1]), 2))
            options["incorrect option 4"] = str(round((distance + abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])) / 3, 2))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata


class PythagoreanTheoremGenerator(GeoPlanGenerator):
    LEG_RANGE = (1, 100)

    schema = {"question_template": "str", "leg1": "str", "leg2": "str"}

    def __init__(
        self,
        metadata: MathTemplateMetaData,
        multiple_choice=True,
        seed=42,
        leg_range=LEG_RANGE,
        num_splices=10,
    ):
        super().__init__(metadata, seed=seed)
        self.leg_range = leg_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        leg1_values = np.linspace(
            self.leg_range[0], self.leg_range[1], self.num_splices)
        leg2_values = np.linspace(
            self.leg_range[0], self.leg_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for leg1 in leg1_values:
                    for leg2 in leg2_values:
                        task_plan = {
                            "question_template": template,
                            "leg1": str(leg1),
                            "leg2": str(leg2),
                        }
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan["question_template"]
        leg1 = round(float(task_plan["leg1"]), 2)
        leg2 = round(float(task_plan["leg2"]), 2)
        hypotenuse = round(np.sqrt(leg1**2 + leg2**2), 2)
        answer = str(hypotenuse)
        question = template.format(leg1=leg1, leg2=leg2)
        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = str(round(leg1 + leg2, 2))
            options["incorrect option 2"] = str(round(np.sqrt(leg1 + leg2), 2))
            options["incorrect option 3"] = str(
                round(np.sqrt((leg1 + leg2) / 2), 2))
            deduped_options = {k: v for i, (k, v) in enumerate(
                options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = str(hypotenuse)
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata
