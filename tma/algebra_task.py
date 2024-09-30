from typing import Dict, List, Tuple
import json
import numpy as np
import sympy

from base import TaskGenerator
from task_store import TaskStore
from utils import *
from metadata import MathTemplateMetaData

class AlgPlanGenerator(TaskGenerator):
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


class PointSlopeGenerator(AlgPlanGenerator):
    COORDINATE_RANGE = (-100, 100)

    schema = {
        'question_template': 'str',
        'x1': 'str',
        'y1': 'str',
        'x2': 'str',
        'y2': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, coordinate_range=COORDINATE_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.coordinate_range = coordinate_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.coordinate_range[0], self.coordinate_range[1], self.num_splices)
        y_values = np.linspace(self.coordinate_range[0], self.coordinate_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                if x1 != x2 or y1 != y2:
                                    task_plan = {
                                        'question_template': template,
                                        'x1': str(x1),
                                        'y1': str(y1),
                                        'x2': str(x2),
                                        'y2': str(y2)
                                    }
                                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        x1 = round(float(task_plan['x1']), 2)
        y1 = round(float(task_plan['y1']), 2)
        x2 = round(float(task_plan['x2']), 2)
        y2 = round(float(task_plan['y2']), 2)
        slope = round((y2 - y1) / (x2 - x1)) if x2 != x1 else None
        question = template.format(x1=x1, y1=y1, x2=x2, y2=y2)
        answer = str(slope) if slope is not None else "undefined"
        
        options = {}
        if self.multiple_choice:
            if slope is not None:
                options["correct"] = str(slope)
                options["incorrect option 1"] = str(-slope)  
                options["incorrect option 2"] = str(round((y2 - y1) / (x2 - x1) / 2, 2))  
                options["incorrect option 3"] = "undefined" if slope != 0 else str(slope)
            else:
                options["correct"] = "undefined"
                options["incorrect option 1"] = "0"  
                options["incorrect option 2"] = str(np.inf)  # Infinite slope 
                options["incorrect option 3"] = str(-np.inf)

            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
        else:
            deduped_options = {}
        
        return question, deduped_options, answer, self.metadata


class RemainderTheoremGenerator(AlgPlanGenerator):
    COEFF_RANGE = (-5, 5)

    schema = {
        'question_template': 'str',
        'polynomial': 'str',
        'root': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, coeff_range=COEFF_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.coeff_range = coeff_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        roots = np.linspace(self.coeff_range[0], self.coeff_range[1], self.num_splices)
        coefficients = np.linspace(self.coeff_range[0], self.coeff_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        x = sympy.symbols('x')
        for _, templates in template_breakdown.items():
            for template in templates:
                for root in roots:
                    for a in coefficients:
                        for b in coefficients:
                            for c in coefficients:
                                root_rounded = round(root, 2)
                                a_rounded = round(a, 2)
                                b_rounded = round(b, 2)
                                c_rounded = round(c, 2)
                                
                                polynomial_expr = a_rounded * x**2 + b_rounded * x + c_rounded
                                polynomial_str = str(polynomial_expr)
                                
                                task_plan = {
                                    'question_template': template,
                                    'polynomial': polynomial_str,
                                    'root': str(root_rounded)
                                }
                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        polynomial_str = task_plan['polynomial']
        root = round(float(task_plan['root']), 2)

        x = sympy.symbols('x')
        polynomial = eval(polynomial_str)
        remainder = round(float(polynomial.subs(x, root)), 2)
        
        question = template.format(polynomial=polynomial_str, root=root)
        answer = str(remainder)
        
        options = {}
        if self.multiple_choice:
            options["correct"] = str(remainder)
            options["incorrect option 1"] = str(round(remainder + np.random.randint(-5, 5) - remainder / 2, 2))
            options["incorrect option 2"] = "0" 
            options["incorrect option 3"] = str(round(remainder - np.random.randint(-5, 5), 2))
            
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
        else:
            deduped_options = {}
        
        return question, deduped_options, answer, self.metadata


class QuadraticFormulaGenerator(AlgPlanGenerator):
    COEFF_RANGE = (-10, 10)

    schema = {
        'question_template': 'str',
        'a': 'str',
        'b': 'str',
        'c': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, coeff_range=COEFF_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.coeff_range = coeff_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        coefficients = np.linspace(self.coeff_range[0], self.coeff_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for a in coefficients:
                    for b in coefficients:
                        for c in coefficients:
                            if a != 0:
                                a_rounded = round(a, 2)
                                b_rounded = round(b, 2)
                                c_rounded = round(c, 2)
                                
                                task_plan = {
                                    'question_template': template,
                                    'a': str(a_rounded),
                                    'b': str(b_rounded),
                                    'c': str(c_rounded)
                                }
                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        a = round(float(task_plan['a']), 2)
        b = round(float(task_plan['b']), 2)
        c = round(float(task_plan['c']), 2)

        discriminant = b**2 - 4*a*c
        x1 = (-b + sympy.sqrt(discriminant)) / (2*a) if discriminant >= 0 else None
        x2 = (-b - sympy.sqrt(discriminant)) / (2*a) if discriminant >= 0 else None

        if x1 is not None and x2 is not None:
            x1 = round(float(x1), 2)
            x2 = round(float(x2), 2)
            answer = f"x1 = {x1}, x2 = {x2}" if x1 != x2 else f"x = {x1}"
        else:
            answer = "No real roots"
        
        question = template.format(a=a, b=b, c=c)
        
        options = {}
        if self.multiple_choice:
            if x1 is not None and x2 is not None:
                options["correct"] = answer
                options["incorrect option 1"] = f"x1 = {x1 + np.random.randint(-5, 5)}, x2 = {x2 + np.random.randint(-5, 5)}"
                options["incorrect option 2"] = "No real roots" 
                options["incorrect option 3"] = f"x1 = {x1 - np.random.randint(-5, 5)}, x2 = {x2 - np.random.randint(-5, 5)}"
            else:
                options["correct"] = "No real roots"
                options["incorrect option 1"] = f"x1 = {np.random.randint(-10, 10)}, x2 = {np.random.randint(-10, 10)}"
                options["incorrect option 2"] = "0"  # Incorrect interpretation
                options["incorrect option 3"] = f"x1 = {np.random.randint(-10, 10)}, x2 = {np.random.randint(-10, 10)}"

            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class ExponentialDecayGenerator(AlgPlanGenerator):
    PARAM_RANGE_N0 = (5, 20)   
    PARAM_RANGE_K = (0.01, 0.5)  
    TIME_RANGE = (0, 3)        

    schema = {
        'question_template': 'str',
        'N0': 'str',
        'k': 'str',
        't': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, param_range_N0=PARAM_RANGE_N0, param_range_k=PARAM_RANGE_K, time_range=TIME_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.param_range_N0 = param_range_N0
        self.param_range_k = param_range_k
        self.time_range = time_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        N0_values = np.linspace(self.param_range_N0[0], self.param_range_N0[1], self.num_splices)
        k_values = np.linspace(self.param_range_k[0], self.param_range_k[1], self.num_splices)
        t_values = np.linspace(self.time_range[0], self.time_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for N0 in N0_values:
                    for k in k_values:
                        for t in t_values:
                            N0_rounded = round(N0, 2)
                            k_rounded = round(k, 2)
                            t_rounded = round(t, 2)
                            
                            task_plan = {
                                'question_template': template,
                                'N0': str(N0_rounded),
                                'k': str(k_rounded),
                                't': str(t_rounded)
                            }
                            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        N0 = round(float(task_plan['N0']), 2)
        k = round(float(task_plan['k']), 2)
        t = round(float(task_plan['t']), 2)
        Nt = round(N0 * np.exp(-k * t), 2)
        
        question = template.format(N0=N0, k=k, t=t)
        answer = str(Nt)
        
        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(round(N0 * np.exp(-k * (t + np.random.uniform(-2, 2))), 2))
            options["incorrect option 2"] = str(round(N0 * (1 - np.exp(-k * t)), 2)) 
            options["incorrect option 3"] = str(round(N0 / (1 + k * t), 2))  # Misinterpretation of decay
            
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata



class PolynomialFactoringGenerator(AlgPlanGenerator):
    COEFF_RANGE = (-10, 10)  

    schema = {
        'question_template': 'str',
        'a': 'str',
        'b': 'str',
        'c': 'str',
        'd': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42, coeff_range=COEFF_RANGE, num_splices=10):
        super().__init__(metadata, seed=seed)
        self.coeff_range = coeff_range
        self.num_splices = num_splices
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        coefficients = np.arange(self.coeff_range[0], self.coeff_range[1] + 1)

        template_breakdown = self.metadata.templates_by_num_params

        for _, templates in template_breakdown.items():
            for template in templates:
                for a in coefficients:
                    for b in coefficients:
                        for c in coefficients:
                            for d in coefficients:
                                if a != 0 and c != 0:
                                    task_plan = {
                                        'question_template': template,
                                        'a': str(a),
                                        'b': str(b),
                                        'c': str(c),
                                        'd': str(d)
                                    }
                                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        a = int(task_plan['a'])
        b = int(task_plan['b'])
        c = int(task_plan['c'])
        d = int(task_plan['d'])

        x = sympy.symbols('x')
        polynomial = sympy.expand((a * x + b) * (c * x + d))
        factored_form = f"({a}x + {b})({c}x + {d})"
        question = template.format(a=a*c, b=a*d + b*c, c=b*d)
        answer = factored_form
        
        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(sympy.factor((a * x + (b + np.random.randint(-2, 3))) * (c * x + d)))
            options["incorrect option 2"] = str(sympy.factor((a * x + b) * (c * x + (d + np.random.randint(-2, 3)))))
            options["incorrect option 3"] = str(sympy.factor((a * x + (b + np.random.randint(-2, 3))) * (c * x + (d + np.random.randint(-2, 3)))))
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata

class MatrixArithmeticGenerator(AlgPlanGenerator):
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
        return json.dumps(task_plan)

    def generate_random_matrix(self):
        return np.random.randint(1, 10, size=self.matrix_size).tolist()

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
            answer = str(result.tolist())
        else :
            result = np.subtract(matrix_a, matrix_b)
            question = task_plan['question_template'].format(matrix_a.tolist(), matrix_b.tolist())
            answer = str(result.tolist())

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


class LinearEquationSystemGenerator(AlgPlanGenerator):
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
        variables = sympy.symbols(f'x1:{self.num_equations + 1}')
        equations = [sum(coefficients[i][j] * variables[j] for j in range(self.num_equations)) - constants[i]
                     for i in range(self.num_equations)]
        solution = sympy.linsolve(equations, *variables)
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
            
            deduped_options = options
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class BasicArithmeticOperationsGenerator(AlgPlanGenerator):
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
                for template_text in templates:
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
            options["incorrect option 1"] = str(integer_1)
            options["incorrect option 3"] = str(integer_2 - integer_1)
            if answer != "undefined":
                options["incorrect option 2"] = str(round(integer_1 / integer_2, 2))
            else:
                options["incorrect option 2"] = str(integer_2)
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata


class RatioTaskGenerator(AlgPlanGenerator):
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
            options["incorrect option 1"] = str(float(answer) + 1)
            options["incorrect option 2"] = str(float(answer) - 1)
            options["incorrect option 3"] = str(float(answer) * 2)
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}
        return question, deduped_options, answer, self.metadata


class SimpleExponentGenerator(AlgPlanGenerator):
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
            options["incorrect option 1"] = str(float(base) ** (float(exponent) + 1))
            options["incorrect option 2"] = str(float(base) ** (float(exponent) - 1))
            options["incorrect option 3"] = "0"
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata
