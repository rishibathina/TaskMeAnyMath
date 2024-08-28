import os
import sys

from typing import Dict, List, Tuple

import numpy as np 
from sympy import symbols, Eq, solve, sqrt
import sympy as sp
from tqdm import tqdm

from baseMath import TaskGenerator
from task_storeMath import TaskStore
from utils import *
import json
import enum
from metadataMath import MathTemplateMetaData

class AlgebraTaskGenerator(TaskGenerator):
    metadata: MathTemplateMetaData

    def __init__(self, metadata: MathTemplateMetaData, seed=42):
        super().__init__(metadata, seed=seed)
    
    def _task_plan_to_str(self, task_plan) -> str:
        "(Abstract method) task plan to string"

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        "(Abstract method) generate task"
        # TODO: Implement the task generation logic here
    
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
    
# Topics:
# 1. matrix problem ( matrix addition, subtraction, )
# 2. solving linear equation systems
# 3. Basic Arithmetic Operations Generator: solving basic addtions, substraction...
# 4. ratio/ portions generator
# 5. exponents

class MatrixProblemGenerator(AlgebraTaskGenerator):
    # Only cares about addition and subtraction first since the templates 
    schema = {
        'question_template': 'str',
        'matrix_a': 'list',
        'matrix_b': 'list',
        'operation': 'str'
    }
    
    def __init__(self, metadata, seed=42, matrix_size=(2, 2), num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
        self.matrix_size = matrix_size
        self.seed = seed
        self.multiple_choice = multiple_choice
        
    def _task_plan_to_str(self, task_plan):
        # convert the task plan into str
        return json.dumps(task_plan)
    
    def generate_random_matrix(self):
        return np.random.randint(1, 10, size=self.matrix_size).tolist()
    
    # enumerate function
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        matrices = [self.generate_random_matrix() for _ in range(self.num_tasks)]
        templates_by_num_params = self.metadata.templates_by_num_params
        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for matrix_a in matrices:
                    for matrix_b in matrices:
                        if "add" in template_text or "Add" in template_text:
                            task_plan = {
                                'question_template': template_text,
                                'matrix_a': matrix_a,
                                'matrix_b': matrix_b,
                                'operation': "addition"
                            }
                            task_store.add(task_plan)
                        elif "subtract" in template_text or "subtraction" in template_text:
                            task_plan = {
                                'question_template': template_text,
                                'matrix_a': matrix_a,
                                'matrix_b': matrix_b,
                                'operation': "subtraction"
                            }
                            task_store.add(task_plan)
                            
    # generate the task
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        matrix_a = np.array(task_plan['matrix_a'])
        matrix_b = np.array(task_plan['matrix_b'])
        operation = task_plan["operation"]
        question = ""
        answer = ""
        options = {}
        if operation == "addition":
            result = np.add(matrix_a, matrix_b)
            question = task_plan['question_template'].format(matrix_a.tolist(), matrix_b.tolist())
            answer = result.tolist()
        else :
            result = np.subtract(matrix_a, matrix_b)
            question = task_plan['question_template'].format(matrix_a.tolist(), matrix_b.tolist())
            answer = result.tolist()
        
        if self.multiple_choice:
            if operation == "addition":
                options["incorrect option 1"] = str((matrix_a + matrix_b * 2).tolist())
                options["incorrect option 2"] = str((matrix_a * 2 + matrix_b).tolist())
                options["incorrect option 3"] = str((matrix_a - matrix_b).tolist())
            elif operation == "subtraction":
                options["incorrect option 1"] = str((matrix_a - matrix_b * 2).tolist())
                options["incorrect option 2"] = str((matrix_a * 2 - matrix_b).tolist())
                options["incorrect option 3"] = str((matrix_a + matrix_b).tolist())
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata
             

# linear equation systems
class LinearEquationSystemGenerator(AlgebraTaskGenerator):
    schema = {
        'question_template': 'str',
        'coefficients': 'list',
        'constants': 'list'
    }
    
    def __init__(self, metadata, seed=42, num_equations=2, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.num_equations = num_equations
        self.seed = seed
        self.multiple_choice = multiple_choice
        
    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)
    
    def generate_random_linear_system(self):
        coefficients = np.random.randint(1, 10, size=(self.num_equations, self.num_equations)).tolist()
        constants = np.random.randint(1, 10, size=self.num_equations).tolist()
        return coefficients, constants

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        for _ in range(self.num_tasks):
            coefficients, constants = self.generate_random_linear_system()
            templates_by_num_params = self.metadata.templates_by_num_params
            for num_params, templates in templates_by_num_params.items():
                for template_text in templates:
                    task_plan = {
                        'question_template': template_text,
                        'coefficients': coefficients,
                        'constants': constants
                    }
                    task_store.add(task_plan)
                    
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        coefficients = np.array(task_plan['coefficients'])
        constants = np.array(task_plan['constants'])
        variables = sp.symbols(f'x1:{self.num_equations + 1}')
        equations = [sum(coefficients[i][j] * variables[j] for j in range(self.num_equations)) - constants[i]
                     for i in range(self.num_equations)]
        solution = sp.linsolve(equations, *variables)
        answer = ""
        options = {}
        question = template.format(coefficients.tolist(), constants.tolist())
        if not solution:
            answer = "No solution found"
        else:
            answer = str(solution)
        if answer == "No solution found":
            if self.multiple_choice:
                options["incorrect option 1"] = "No solution exists"
                options["incorrect option 2"] = "Infinite solutions"
                options["incorrect option 3"] = "Dependent system"
                deduped_options = options
                deduped_options["correct"] = answer
            else:
                deduped_options = {}
            return question, deduped_options, answer, self.metadata
        solution = list(solution)[0]
        if self.multiple_choice:
            options["incorrect option 1"] = str(tuple(np.array(solution) + np.random.randint(3, 6, size=self.num_equations)))
            while options["incorrect option 1"] == answer:
                options["incorrect option 1"] = str(tuple(np.array(solution) + np.random.randint(3, 6, size=self.num_equations)))
            options["incorrect option 2"] = str(tuple(np.array(solution) + np.random.randint(-4, -1, size=self.num_equations)))
            while options["incorrect option 2"] == answer or options["incorrect option 2"] == options["incorrect option 1"]:
                options["incorrect option 2"] = str(tuple(np.array(solution) + np.random.randint(-4, -1, size=self.num_equations)))
            options["incorrect option 3"] = str(tuple(np.array(solution) + np.random.randint(1, 3, size=self.num_equations)))
            while options["incorrect option 3"] == answer or options["incorrect option 3"] == options["incorrect option 1"] or options["incorrect option 3"] == options["incorrect option 2"]:
                options["incorrect option 3"] = str(tuple(np.array(solution) + np.random.randint(1, 3, size=self.num_equations)))
            # Deduplicate options
            deduped_options = options
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata
    
# Basic Arithmetic Operations Generator: solving basic addtions, substraction...

class BasicArithmeticOperationsGenerator(AlgebraTaskGenerator):
    schema = {
        'question_template': 'str',
        'operands': 'list',
        'operation': 'str'
    }
    
    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice
        
    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store, operand_range=(1, 100)):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params
        for _ in range(self.num_tasks):
            integer_1 = np.random.randint(*operand_range)
            integer_2 = np.random.randint(*operand_range)
            for num_params, templates in templates_by_num_params.items():
                for template_text in tqdm(templates, desc=f"Enumerating templates with {num_params} params"):
                    operation = ""
                    if "+" in template_text or "add" in template_text:
                        operation = "addition"
                    elif "subtract" in template_text or "-" in template_text:
                        operation = "subtraction"
                    elif "product" in template_text:
                        operation = "multiplication"
                    elif "divided" in template_text or "/" in template_text:
                        operation = "division"
                    task_plan = {
                        'question_template': template_text,
                        'operands': [integer_1, integer_2],
                        'operation': operation
                    }
                    task_store.add(task_plan)
                    
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        integer_1, integer_2 = task_plan['operands']
        operation = task_plan['operation']
        answer = ""
        options = {}
        if operation == "addition":
            result = integer_1 + integer_2
            answer = str(result)
        elif operation == "subtraction":
            result = integer_1 - integer_2
            answer = str(result)
        elif operation == "multiplication":
            result = integer_1 * integer_2
            answer = str(result)
        elif operation == "division":
            if integer_2 == 0:
                answer = "undefined"
            else:
                result = integer_1 / integer_2
                answer = f"{result:.2f}"
        question = template.format(integer_1, integer_2)
        if self.multiple_choice:
            if answer == "undefined":
                options["incorrect option 1"] = integer_1
                options["incorrect option 2"] = integer_2
                options["incorrect option 3"] = integer_1 + integer_2
            else:
                options["incorrect option 1"] = float(answer) + 1
                options["incorrect option 2"] = float(answer) - 1
                options["incorrect option 3"] = float(answer) * 2
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata
    
# 4. ratio/ portions generator
class RatioTaskGenerator(AlgebraTaskGenerator):
    schema = {
        'question_template': 'str',
        'ratio1': 'int',
        'ratio2': 'int',
        'base_quantity': 'int'
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice
        
    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params
        for _ in range(self.num_tasks):
            ratio1 = np.random.randint(1, 10)
            ratio2 = np.random.randint(1, 10)
            base_quantity = np.random.randint(5, 50)
            for num_params, templates in templates_by_num_params.items():
                for template_text in templates:
                    task_plan = {
                        'question_template': template_text,
                        'ratio1': ratio1,
                        'ratio2': ratio2,
                        'base_quantity': base_quantity
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        ratio1 = task_plan['ratio1']
        ratio2 = task_plan['ratio2']
        base_quantity = task_plan['base_quantity']

        calculated_value = base_quantity * (ratio2 / ratio1)
        question = template.format(ratio1, ratio2, base_quantity)
        answer = f"{calculated_value:.2f}"  # Format to two decimal places

        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = float(answer) + 1
            options["incorrect option 2"] = float(answer) - 1
            options["incorrect option 3"] = float(answer) * 2
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata
    
    
# 5. simple exponents problems
class SimpleExponentGenerator(AlgebraTaskGenerator):
    schema = {
        'question_template': 'str',
        'base': 'str',
        'exponent': 'str'
    }
    
    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.seed = seed
        self.num_tasks = num_tasks
        self.multiple_choice = multiple_choice
        np.random.seed(seed)
        
    def _task_plan_to_str(self, task_plan):
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params
        for _ in range(self.num_tasks):
            base = np.random.randint(2, 10)
            exponent = np.random.randint(2, 6)
            for num_params, templates in templates_by_num_params.items():
                for template_text in templates:
                    task_plan = {
                        'question_template': template_text,
                        'base': base,
                        'exponent': exponent
                    }
                    task_store.add(task_plan)
                    
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        base = task_plan['base']
        exponent = task_plan['exponent']

        calculated_value = float(base) ** float(exponent)
        question = template.format(base, exponent)
        answer = str(calculated_value)

        options = {}
        if self.multiple_choice:
            options["incorrect option 1"] = float(base) ** (float(exponent) + 1)
            options["incorrect option 2"] = float(base) ** (float(exponent) - 1)
            options["incorrect option 3"] = 0
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        
        return question, deduped_options, answer, self.metadata
    
    
    

    

