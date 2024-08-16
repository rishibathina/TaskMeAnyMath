import os
import sys

from typing import Dict, List, Tuple

import numpy as np 
from tqdm import tqdm

from baseMath import TaskGenerator
from task_storeMath import TaskStore
from utils import *
import json
import enum
from metadataMath import MathTemplateMetaData
# PERI_SIDE_ONE_RANGE, PERI_SIDE_TWO_RANGE, PERI_SIDE_THREE_RANGE, 
#     make_single_prod, make_pair_prod, make_triplet_prod

class Template(enum.Enum):
    PERIMETER_TEMPLATES = 'MathVerse/math_anotations/perimeter_templates.json'
    MIDPOINT_TEMPLATES = 'annotations/math_annotations/midpoint.json'

class GeoPlanGenerator(TaskGenerator):
    metadata: MathTemplateMetaData

    def __init__(self, metadata: MathTemplateMetaData, seed=42):
        super().__init__(metadata, seed=seed)
    
    def _task_plan_to_str(self, task_plan) -> str:
        "(Abstract method) task plan to string"

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        "(Abstract method) generate task"
        # TODO: COME BACK IN FILL THIS IN AS NOT A ABSTRACT METHOD
        
    
    def generate(self, task_plan, return_data=True, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        question, options, answer, math_metadata = self._generate_task(task_plan)

        task = {
            'question'  : question.replace('_', ' '),
            'options'   : options if self.multiple_choice else None,
            'answer'    : answer.replace('_', ' '),
            'task_plan' : self._task_plan_to_str(task_plan),
            'math_metadata' : math_metadata,
            
        }

        return task

    
class PerimeterGenerator(GeoPlanGenerator):
    schema = {
        'question_template' : 'str',
        'side_one'          : 'str',
        'side_two'          : 'str',
        'side_three'        : 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=False, seed=42, side_one_range=PERI_SIDE_ONE_RANGE, side_two_range=PERI_SIDE_TWO_RANGE, side_three_range=PERI_SIDE_THREE_RANGE, num_splices=3):
        super().__init__(metadata, seed=seed)
        self.side_one_range = side_one_range
        self.side_two_range = side_two_range
        self.side_three_range = side_three_range
        self.int_to_peri_list = int_to_peri_list
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        single = make_single_prod(self.side_one_range, self.num_splices)
        pairs = make_pair_prod(self.side_one_range, self.side_three_range, self.num_splices)
        triplets = make_triplet_prod(self.side_one_range, self.side_two_range, self.side_three_range, self.num_splices)
        
        template_breakdown = self.metadata.templates_by_num_params
        
        for param_count, templates in template_breakdown.items():
            peri_list = locals()[self.int_to_peri_list[param_count]]
            
            for template in tqdm(templates, desc=f"Enumerating templates with {param_count} params"):
                for group in peri_list:
                    if isinstance(group, (float, np.float64)):  # Check if group is a float
                        group = [group]
                    params = [str(param) if param is not None else None for param in group]                   
                    while len(params) < 3:  
                        params.append(None)

                    task_plan = {
                        'question_template': template,
                        'side_one': params[0],
                        'side_two': params[1],
                        'side_three': params[2]
                    }
                    
                    task_store.add(task_plan)
                
            
    def _generate_task(self, task_plan) -> Tuple[str | List[str] | Dict]:
        question = None
        answer = None

        template = task_plan['question_template']
        side_one = float(task_plan['side_one'])
        side_two = float(task_plan['side_two']) if task_plan['side_two'] is not None else None
        side_three = float(task_plan['side_three']) if task_plan['side_three'] is not None else None

        options = []
        if side_two is None:
            question = template.format(side_one = side_one) # format is single param
            answer = str(3 * side_one)
            if self.multiple_choice:
                options.append(2 * side_one)
                options.append(side_one)
                options.append(side_one * side_one / 2)
            
        elif side_three is None:
            question = template.format(side_one=side_one, side_two=side_two) # format is double param
            answer = str(2 * side_one + side_two)
            if self.multiple_choice:
                options.append(3 * side_one)
                options.append(side_one + (2 * side_two))
                options.append((side_one * side_two) / 2)
        
        else:
            question = template.format(side_one=side_one, side_two=side_two, side_three=side_three) # format is triple param
            answer = str(side_one + side_two + side_three)
            if self.multiple_choice:
                options.append(3 * side_one)
                options.append(side_one + (2 * side_two))
                options.append((side_one * side_two) / 2)
        
        
            
        return question, options, answer, self.metadata

    
class MidpointGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 100)
    Y_COORDINATE_RANGE = (0, 100)
    Z_COORDINATE_RANGE = (0, 100)

    schema = {
        'question_template': 'str',
        'point1': 'str',
        'point2': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, z_range=Z_COORDINATE_RANGE, num_splices=5):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)
        z_values = np.linspace(self.z_range[0], self.z_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for midpoint tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for z1 in z_values:
                            for x2 in x_values:
                                for y2 in y_values:
                                    for z2 in z_values:
                                        task_plan = {
                                            'question_template': template,
                                            'point1': (x1, y1, z1),
                                            'point2': (x2, y2, z2)
                                        }
                                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        point1 = eval(task_plan['point1'])
        point2 = eval(task_plan['point2'])

        midpoint = (
            (point1[0] + point2[0]) / 2,
            (point1[1] + point2[1]) / 2,
            (point1[2] + point2[2]) / 2
        )
        
        question = template.format(
            param1 = point1,
            param2 = point2
        )
        correct_answer = "({}, {}, {})".format(midpoint[0], midpoint[1], midpoint[2])
        
        options = []
        if self.multiple_choice:
            # Correct answer
            options.append(correct_answer)
            options.append("({}, {}, {})".format(
                point1[0] + point2[0], point1[1] + point2[1], point1[2] + point2[2]  # Adding instead of averaging
            ))
            options.append("({}, {}, {})".format(
                midpoint[0], (point1[1] + point2[1]) / 3, midpoint[2]  # Incorrect division
            ))
            options.append("({}, {}, {})".format(
                (point1[0] + point2[0]) / 2, midpoint[2], midpoint[1]  # Swapping y and z in the midpoint calculation
            ))
            np.random.shuffle(options)
        
        return question, options, correct_answer, self.metadata
    

class IntersectionGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 10) 
    Y_COORDINATE_RANGE = (0, 10)

    schema = {
        'question_template': 'str',
        'vector1_start': 'str',
        'vector1_end': 'str',
        'vector2_start': 'str',
        'vector2_end': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, num_splices=4):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for intersection tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                for x3 in x_values:
                                    for y3 in y_values:
                                        for x4 in x_values:
                                            for y4 in y_values:
                                                task_plan = {
                                                    'question_template': template,
                                                    'vector1_start': str((x1, y1)),
                                                    'vector1_end': str((x2, y2)),
                                                    'vector2_start': str((x3, y3)),
                                                    'vector2_end': str((x4, y4))
                                                }
                                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        vector1_start = eval(task_plan['vector1_start'])
        vector1_end = eval(task_plan['vector1_end'])
        vector2_start = eval(task_plan['vector2_start'])
        vector2_end = eval(task_plan['vector2_end'])

        vector1 = np.array(vector1_end) - np.array(vector1_start)
        vector2 = np.array(vector2_end) - np.array(vector2_start)

        if np.cross(vector1, vector2) == 0:
            answer = "null"
            options = ["null"]
        else:
            # Compute intersection
            A = np.array([[vector1[0], -vector2[0]], [vector1[1], -vector2[1]]])
            b = np.array([vector2_start[0] - vector1_start[0], vector2_start[1] - vector1_start[1]])
            t = np.linalg.solve(A, b)
            intersection = vector1_start + t[0] * vector1
            answer = str(tuple(intersection))

            if self.multiple_choice:
                options = [answer]
                options.append(str(vector1_start))  
                options.append(str(vector1_end))    
                options.append(str(vector2_start))  
                options.append(str(vector2_end))    

                options = list(dict.fromkeys(options)) 
                options = options[:4] 
                np.random.shuffle(options)
            else:
                options = []

        question = template.format(
            param1=vector1_start,
            param2=vector1_end,
            param3=vector2_start,
            param4=vector2_end
        )
        
        return question, options, answer, self.metadata



class PerpendicularGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 10) 
    Y_COORDINATE_RANGE = (0, 10)

    schema = {
        'question_template': 'str',
        'vector1_start': 'str',
        'vector1_end': 'str',
        'vector2_start': 'str',
        'vector2_end': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, num_splices=4):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for perpendicular tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                for x3 in x_values:
                                    for y3 in y_values:
                                        for x4 in x_values:
                                            for y4 in y_values:
                                                task_plan = {
                                                    'question_template': template,
                                                    'vector1_start': str((x1, y1)),
                                                    'vector1_end': str((x2, y2)),
                                                    'vector2_start': str((x3, y3)),
                                                    'vector2_end': str((x4, y4))
                                                }
                                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        template = task_plan['question_template']
        vector1_start = eval(task_plan['vector1_start'])
        vector1_end = eval(task_plan['vector1_end'])
        vector2_start = eval(task_plan['vector2_start'])
        vector2_end = eval(task_plan['vector2_end'])

        vector1 = np.array(vector1_end) - np.array(vector1_start)
        vector2 = np.array(vector2_end) - np.array(vector2_start)

        dot_product = np.dot(vector1, vector2)
        if dot_product == 0:
            answer = "Yes, perpendicular"
        else:
            answer = "Not perpendicular"

        question = template.format(
            param1=vector1_start,
            param2=vector1_end,
            param3=vector2_start,
            param4=vector2_end
        )
        
        return question, answer, self.metadata


class SideLengthGenerator(GeoPlanGenerator):
    AREA_RANGE = (1, 100)

    schema = {
        'question_template': 'str',
        'area': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, area_range=AREA_RANGE, num_splices=5):
        super().__init__(metadata, seed=seed)
        self.area_range = area_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        area_values = np.linspace(self.area_range[0], self.area_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for square side length tasks"):
                for area in area_values:
                    task_plan = {
                        'question_template': template,
                        'area': str(area)
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        area = float(task_plan['area'])

        side_length = np.sqrt(area)

        question = template.format(area=area)
        answer = str(side_length)

        options = []
        if self.multiple_choice:
            options.append(answer)
            options.append(str(area / 2))  # Dividing area by 2
            options.append(str(side_length * 2)) 
            options.append(str(np.sqrt(area / 2))) #Using half the area

            options = list(dict.fromkeys(options))
            options = options[:4] 
            np.random.shuffle(options)
        
        return question, options, answer, self.metadata

        

class ArcLengthGenerator(GeoPlanGenerator):
    RADIUS_RANGE = (1, 10) 
    ANGLE_RANGE = (0, 360)

    schema = {
        'question_template': 'str',
        'radius': 'str',
        'angle': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, radius_range=RADIUS_RANGE, angle_range=ANGLE_RANGE, num_splices=5):
        super().__init__(metadata, seed=seed)
        self.radius_range = radius_range
        self.angle_range = angle_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        radius_values = np.linspace(self.radius_range[0], self.radius_range[1], self.num_splices)
        angle_values = np.linspace(self.angle_range[0], self.angle_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for arc length tasks"):
                for radius in radius_values:
                    for angle in angle_values:
                        task_plan = {
                            'question_template': template,
                            'radius': str(radius),
                            'angle': str(angle)
                        }
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        radius = float(task_plan['radius'])
        angle = float(task_plan['angle'])

        arc_length = (np.pi * radius * angle) / 180

        question = template.format(radius=radius, angle=angle)
        answer = str(arc_length)
        
        options = []
        if self.multiple_choice:
            options.append(answer)
            options.append(str(2 * np.pi * radius * (angle / 360))) 
            options.append(str(np.pi * radius * angle / 360))  # wrong denominator for degrees
            options.append(str(np.pi * radius**2 * angle / 180))  # area of a sector
            options.append(str(np.pi * angle / 180))
            
            options = list(dict.fromkeys(options))
            np.random.shuffle(options)
        
        return question, options, answer, self.metadata



class CircleGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'radius': 'str',          # Store float as string
        'circumference': 'str',   # Store float as string
        'area': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store: TaskStore):
        all_values = np.linspace(1, 100, self.num_tasks)  # general range for any circle property
        template_breakdown = self.metadata.templates_by_num_params
        for num_params, texts in template_breakdown.items():
            for text in texts:
                for value in tqdm(all_values, desc=f"Enumerating templates with {num_params} params"):
                    if 'circumference' in text or 'perimeter' in text:
                        radius = value / (2 * np.pi)
                        circumference = value
                        area = np.pi * radius ** 2
                    elif 'area' in text:
                        radius = np.sqrt(value / np.pi)
                        circumference = 2 * np.pi * radius
                        area = value
                    else:  # default to radius if not specified
                        radius = value
                        circumference = 2 * np.pi * radius
                        area = np.pi * radius ** 2
                    
                    result = (radius, circumference, area)
                    if result not in self.all_tasks:
                        task_plan = {
                            'question_template': text,
                            'radius': str(radius),
                            'circumference': str(circumference),
                            'area': str(area)
                        }
                        task_store.add(task_plan)
                        self.all_tasks.add(result)

    def _generate_task(self, task_plan) -> Tuple[str, str, Dict]:
        answer = None
        question = None
        template = task_plan['question_template']
        radius = float(task_plan['radius'])
        circumference = float(task_plan['circumference'])
        area = float(task_plan['area'])
        options = []
        
        if 'circumference' in template or 'perimeter' in template:
            if 'radius' in template:
                question = template.format(param1=radius)
                # in this case, we are looking for circumference, known radius
                # the correct formula would be circumference = 2 * np.pi * radius
                answer = circumference
                if self.multiple_choice:
                    option1 = np.pi * radius  # forget the 2
                    option2 = 0.5 * np.pi * radius
                    option3 = np.pi * radius ** 2  # think it's area calculation
                    options.append(option1)
                    options.append(option2)
                    options.append(option3)
            else:
                question = template.format(param1=circumference)
                answer = radius
                # in this case, we know circumference and try to know the radius
                # correct formula to calculate radius would be circumference/ (2 * np.pi)
                if self.multiple_choice:
                    option1 =  circumference/ (np.pi)
                    option2 =  circumference/ (0.5 * np.pi)
                    option3 =  np.sqrt(circumference / np.pi)
                    options.append(option1)
                    options.append(option2)
                    options.append(option3) 
        elif 'area' in template:
            if 'radius' in template:
                question = template.format(param1=radius)
                answer = area
                # known radius, calculate area
                # the correct formula would be np.pi * radius ** 2
                if self.multiple_choice:
                    option1 = np.pi * radius
                    option2 = np.pi * radius ** 3
                    option3 = 2 * np.pi * radius
                    options.append(option1)
                    options.append(option2)
                    options.append(option3)
            else:
                question = template.format(param1=area)
                answer = radius
                # known area, calculate the radius
                # correct formula here is np.sqrt(area / np.pi)
                if self.multiple_choice:
                    option1 = area / np.pi
                    option2 = 2 * np.sqrt(area / np.pi)
                    option3 = area/ (2 * np.pi)
                    options.append(option1)
                    options.append(option2)
                    options.append(option3)
        else:
            raise ValueError("Template must specify either perimeter, area, or radius.")
        np.random.shuffle(options)
        return question, str(answer), options, self.metadata
    

class TriangleAreaGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'base': 'str',
        'height': 'str'
    }
    
    def __init__(self, metadata: MathTemplateMetaData, seed=42, base_range=(1, 100), height_range=(1, 100), num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.base_range = base_range
        self.height_range = height_range
        self.multiple_choice = multiple_choice
    
    
    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store: TaskStore):
        # this method should put all possible tasks to the taskstore
        template_breakdown = self.metadata.templates_by_num_params
        all_bases = np.linspace(self.base_range[0], self.base_range[1], self.num_tasks)
        all_heights = np.linspace(self.height_range[0], self.height_range[1], self.num_tasks)
        for num_params, templates in template_breakdown.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for base in all_bases:
                    for height in all_heights:
                        task_plan = {
                                'question_template': template_text,
                                'base': str(base),
                                'height': str(height)
                            }
                        task_store.add(task_plan)
                        self.all_tasks.add((base, height))
        
    def _generate_task(self, task_plan) -> Tuple[str, str, Dict]:
        # generates single task
        question = task_plan['question_template']
        base = task_plan['base']
        height = task_plan["height"]
        options = []
        area = 0.5 * float(base) * float(height)
        if (self.multiple_choice) :
            # come up with three possible wrong answer
            # 1: does not times 0.5
            option1 = float(base) * float(height)
            # 2: times 0.25 instead of 0.5
            option2 = 0.25 * float(base) * float(height)
            # 3: calculate base * base
            option3 = float(base) * float(base)
            options.append(option1)
            options.append(option2)
            options.append(option3)
        question_text = question.format(base, height)
        area = str(area)
        np.random.shuffle(options)
        return question_text, area, options, self.metadata


class AngleSumGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'shape': 'str',
        'sides': 'int'
    }
    
    def __init__(self, metadata, seed=42, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.all_tasks = set()
        # store possible shapes
        self.shapes = {
            'triangle': 3,
            'quadrilateral': 4,
            'pentagon': 5,
            'hexagon': 6,
            'heptagon': 7,
            'octagon': 8,
            'nonagon': 9,
            'decagon': 10
        }
        self.multiple_choice = multiple_choice
        
         
    def _task_plan_to_str(self, task_plan):
        # rgus method converts taskplan into str
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store):
        # put all the tasks into the task store
        templates_by_num_params = self.metadata.templates_by_num_params
        for num_params, templates in templates_by_num_params.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for shape, sides in self.shapes.items():
                    task_plan = {
                            'question_template': template_text,
                            'shape': shape,
                            'sides': sides
                        }
                    task_store.add(task_plan)
        
    def _generate_task(self, task_plan):
        # calculate the angle sum of the problem
        # generate one task
        # remember to avoid regenerating same problem here
        template = task_plan['question_template']
        shape = task_plan['shape']
        side = task_plan['sides']
        problem = (shape, side)
        self.all_tasks.add(problem)
        angle_sum = (side - 2) * 180
        question = template.format(shape)
        answer = str(angle_sum)
        options = []
        if self.multiple_choice:
            while len(options) < 3:
                wrong_shape = np.random.choice(list(self.shapes.keys()))
                wrong_sides = self.shapes[wrong_shape]
                wrong_angle_sum = (wrong_sides - 2) * 180
                if wrong_angle_sum != angle_sum and wrong_angle_sum not in options:
                    options.append(wrong_angle_sum)
        np.random.shuffle(options)
        return question, answer, options, self.metadata


class VolumeRectangularPrismGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'length': 'str',
        'width': 'str',
        'height': 'str'
    }

    def __init__(self, metadata, seed=42, length_range=(1, 10), width_range=(1, 10), height_range=(1, 10), num_tasks=100, multi_options = False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.length_range = length_range
        self.width_range = width_range
        self.height_range = height_range
        self.seed = seed
        self.multi_options = multi_options
        

    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        lengths = np.random.uniform(self.length_range[0], self.length_range[1], self.num_tasks)
        widths = np.random.uniform(self.width_range[0], self.width_range[1], self.num_tasks)
        heights = np.random.uniform(self.height_range[0], self.height_range[1], self.num_tasks)
        templates_by_num_params = self.metadata.templates_by_num_params
        for num_params, templates in templates_by_num_params.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for length in lengths:
                    for width in widths:
                        for height in heights:
                            task_plan = {
                            'question_template': template_text,
                            'length': length,
                            'width': width,
                            'height': height
                            }
                            task_store.add(task_plan)
    
    def _generate_task(self, task_plan):
        template = task_plan['question_template']
        height = float(task_plan['height'])
        width = float(task_plan['width'])
        length = float(task_plan['length'])
        problem = (height, width, length)
        self.all_tasks.add(problem)
        options = []
        volume = length * width * height
        if (self.multi_options):
            option1 = length * width * 2
            option2 = length * length * length
            option3 = length + width + height
            options.append(option1)
            options.append(option2)
            options.append(option3)
        question = template.format(length, width, height)
        answer = str(volume)
        np.random.shuffle(options)
        return question, answer, options, self.metadata
        

class AngleGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'angles': 'list'  # Store list of angles
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, num_tasks=100, multiple_choice = False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.multiple_choice = multiple_choice
    
    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store: TaskStore):
        # store tasks in the task store
        template_breakdown = self.metadata.templates_by_num_params
        shapes = {
            'equilateral triangle': 60,
            'rectangle': 90,
            'regular pentagon': 108,
            'regular hexagon': 120
        }
        for num_params, templates in template_breakdown.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for shape, angles in shapes.items():
                    task_plan = {
                        'question_template': template_text,
                        'angles': angles
                    }
                    task_store.add(task_plan)
        
    def _generate_task(self, task_plan) -> Tuple[str, str, Dict]:
        # generate the single task
        question = None
        answer = None

        template = task_plan['question_template']
        angles = task_plan['angles']
        options = []
        if angles == 60:
            answer = "equilateral triangle"
            if self.multiple_choice:
                options.append("rectangle")
                options.append("regular pentagon")
                options.append("regular hexagon")
        elif angles == 90:
            answer = "rectangle"
            if self.multiple_choice:
                options.append("equilateral triangle")
                options.append("regular pentagon")
                options.append("regular hexagon")
        elif angles == 108:
            answer = "regular pentagon"
            if self.multiple_choice:
                options.append("equilateral triangle")
                options.append("rectangle")
                options.append("regular hexagon")
        elif angles == 120:
            answer = "regular hexagon"
            if self.multiple_choice:
                options.append("equilateral triangle")
                options.append("rectangle")
                options.append("regular pentagon")
        else:
            answer = "unreasonable shape"

        question = template.format(angles)
        np.random.shuffle(options)
        return question, answer, options, self.metadata
    

class VolumeSphereGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'radius': 'str'
    }

    def __init__(self, metadata, seed=42, radius_range=(1, 20), num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.radius_range = radius_range
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Serialize task plan for uniqueness checking
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        radii = np.random.uniform(self.radius_range[0], self.radius_range[1], self.num_tasks)
        templates_by_num_params = self.metadata.templates_by_num_params
        for num_params, templates in templates_by_num_params.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for radius in radii:
                    task_plan = {
                        'question_template': template_text,
                        'radius': radius
                    }
                    if self._task_plan_to_str(task_plan) not in self.all_tasks:
                        self.all_tasks.add(self._task_plan_to_str(task_plan))
                        task_store.add(task_plan)
    
    def _generate_task(self, task_plan):
        template = task_plan['question_template']
        radius = float(task_plan['radius'])
        volume = (4/3) * np.pi * (radius ** 3)
        options = []
        question = template.format(radius)
        answer = f"{volume:.2f} cubic units"
        if self.multiple_choice:
            option1 = np.pi * (radius ** 3)
            option2 = (1/3) * np.pi * (radius ** 3)
            option3 = (4/3) * np.pi * (radius ** 2)
            options.append(option1)
            options.append(option2)
            options.append(option3)
        np.random.shuffle(options)
        return question, answer, options, self.metadata


class ConeVolumeGenerator:
    schema = {
        'question_template': 'str',
        'radius': 'str',
        'height': 'str'
    }

    def __init__(self, metadata, seed=42, radius_range=(1, 20), height_range=(1, 20), num_tasks=100, multiple_choice=False):
        self.metadata = metadata
        self.seed = seed
        self.radius_range = radius_range
        self.height_range = height_range
        self.num_tasks = num_tasks
        self.all_tasks = set()
        np.random.seed(self.seed)
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Serialize task plan for uniqueness checking
        return json.dumps(task_plan, sort_keys=True)

    def enumerate_task_plans(self, task_store):
        radii = np.random.uniform(self.radius_range[0], self.radius_range[1], self.num_tasks)
        heights = np.random.uniform(self.height_range[0], self.height_range[1], self.num_tasks)
        templates_by_num_params = self.metadata.templates_by_num_params
        for num_params, templates in templates_by_num_params.items():
            for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                for radius in radii:
                    for height in heights:
                        task_plan = {
                            'question_template': template_text,
                            'radius': radius,
                            'height': height
                        }
                        task_plan_str = self._task_plan_to_str(task_plan)
                        if task_plan_str not in self.all_tasks:
                            self.all_tasks.add(task_plan_str)
                            task_store.add(task_plan)

    def _generate_task(self, task_plan):
        template = task_plan['question_template']
        radius = float(task_plan['radius'])
        height = float(task_plan['height'])
        volume = (1/3) * np.pi * (radius ** 2) * height
        options = []
        if self.multiple_choice:
            option1 = (4/3) * np.pi * (radius ** 3)
            option2 = (4/3) * np.pi * (radius ** 2) * height
            option3 = np.pi * (radius ** 2) * height
            options.append(option1)
            options.append(option2)
            options.append(option3)
        question = template.format(radius, height)
        answer = f"{volume:.2f} cubic cm"
        np.random.shuffle(options)
        return question, answer, options, self.metadata


class PointDistanceGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (-10, 10)
    Y_COORDINATE_RANGE = (-10, 10)

    schema = {
        'question_template': 'str',
        'point1': 'str',
        'point2': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for distance tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                if (x1, y1) != (x2, y2): 
                                    task_plan = {
                                        'question_template': template,
                                        'point1': str((x1, y1)),
                                        'point2': str((x2, y2))
                                    }
                                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        point1 = eval(task_plan['point1'])
        point2 = eval(task_plan['point2'])

        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        
        question = template.format(point1=point1, point2=point2)
        answer = str(distance)
        
        options = []
        if self.multiple_choice:
            options.append(str(distance))
            options.append(str(abs(point2[0] - point1[0]) + abs(point2[1] - point1[1]))) 
            options.append(str(abs(point2[0] - point1[0])))  # Horizontal distance
            options.append(str(abs(point2[1] - point1[1])))  # Vertical distance 
            options.append(str((distance + abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])) / 3))
            options = list(dict.fromkeys(options))  
            np.random.shuffle(options)
        
        return question, options, answer, self.metadata        


class PythagoreanTheoremGenerator(GeoPlanGenerator):
    LEG_RANGE = (1, 100)

    schema = {
        'question_template': 'str',
        'leg1': 'str',
        'leg2': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, leg_range=LEG_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.leg_range = leg_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        leg1_values = np.linspace(self.leg_range[0], self.leg_range[1], self.num_splices)
        leg2_values = np.linspace(self.leg_range[0], self.leg_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for Pythagorean theorem tasks"):
                for leg1 in leg1_values:
                    for leg2 in leg2_values:
                        task_plan = {
                            'question_template': template,
                            'leg1': str(leg1),
                            'leg2': str(leg2)
                        }
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, List[str], str, Dict]:
        template = task_plan['question_template']
        leg1 = float(task_plan['leg1'])
        leg2 = float(task_plan['leg2'])

        hypotenuse = np.sqrt(leg1**2 + leg2**2)
        
        question = template.format(leg1=leg1, leg2=leg2)
        answer = str(hypotenuse)
        
        options = []
        if self.multiple_choice:
            options.append(str(hypotenuse))
            options.append(str(leg1 + leg2))  # Adding the legs directly
            options.append(str(np.sqrt(leg1 + leg2)))  # Failed to sqaure legs
            options.append(str(np.sqrt((leg1 + leg2) / 2)))  # Average of the legs
            #np.random.shuffle(options)
        
        return question, options, answer, self.metadata
