from typing import Dict, List, Tuple  # Import the necessary typing tools only once
import json  # For handling JSON data
import numpy as np  # For random number generation
import sympy  # For symbolic math
import random  # For generating random choices in multiple choice options
import re
# Importing project-specific modules
from task_store import TaskStore  # Assuming TaskStore is needed for task storage
from utils import *  # Assuming utility functions are imported from utils.py
from metadata import MathTemplateMetaData  # Assuming this handles the metadata for math templates

class TaskGenerator:
    schema = {}

    def __init__(self, metadata: MathTemplateMetaData, seed=42):
        self.metadata = metadata
        self.rng = np.random.default_rng(seed=seed)

    def _compose_options(self, answer, negatives):
        # Ensure negatives have the correct number of options
        if len(negatives) > NUM_OPTIONS - 1:
            negatives = self.rng.choice(negatives, NUM_OPTIONS - 1, replace=False).tolist()
        options = [answer] + negatives
        return options

    def _task_plan_to_str(self, task_plan) -> str:
        """
        (Abstract method) convert task plan to string for task embedding.
        Override in subclasses to implement specific logic.
        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def enumerate_task_plans(self, task_store: TaskStore):
        """
        (Abstract method) enumerate task plans.
        Override in subclasses to implement specific logic.
        """
        raise NotImplementedError("This method needs to be implemented by a subclass")

    def generate(self, task_plan, return_data=True, seed=None):
        """
        (Abstract method) generate task based on task plan.
        Override in subclasses to implement specific logic.
        """
        raise NotImplementedError("This method needs to be implemented by a subclass")
  

class CalcPlanGenerator(TaskGenerator):
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


class DifferentiationGenerator(CalcPlanGenerator):
    schema = {
        'question_template': 'str',
        'operands': 'list',
        'operation': 'str',
        'derivative_str':'int'
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        self.metadata = metadata
        self.seed = seed
        self.num_tasks = num_tasks
        self.multiple_choice = multiple_choice
        random.seed(seed)
        
        # Load templates from metadata
        self.templates_by_num_params = self.metadata.templates_by_num_params

    def generate_polynomial_term(self, variable: str, degree: int) -> str:
        """Generate only x^2 or x^3 terms with random coefficients."""
        coefficient = random.randint(1, 10)  # Random coefficient between 1 and 10
        if degree == 2 or degree == 3:
            return f"{coefficient}{variable}^{degree}"  # Limited to x^2 or x^3 terms

    def differentiate_polynomial_term(self, term: str, variable: str) -> str:
        """Differentiate only terms of the form ax^2 or ax^3."""
        if "^2" in term:
            coefficient = int(term.split(variable)[0])
            return f"{2 * coefficient}{variable}"  # Derivative of ax^2 is 2ax
        elif "^3" in term:
            coefficient = int(term.split(variable)[0])
            return f"{3 * coefficient}{variable}^2"  # Derivative of ax^3 is 3ax^2
        return "0"

    def enumerate_task_plans(self, task_store, variable="x"):
        """Generate tasks involving only x^2 or x^3 terms and store them."""
        for _ in range(self.num_tasks):
            degree = random.choice([2, 3])
            term = self.generate_polynomial_term(variable, degree)
            derivative = self.differentiate_polynomial_term(term, variable)

            # Pick a random template
            template_key = random.choice(list(self.templates_by_num_params.keys()))
            template_text = self.templates_by_num_params[template_key]

            task_plan = {
                'question_template': template_text,
                'operands': [term],
                'operation': 'differentiation',
                'derivative_str': derivative
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        """Generate a single task from a task plan."""
        template = task_plan['question_template']
        polynomial = task_plan['operands'][0]

        # Format the question using the template
        question = template.format(polynomial)
        
        # We assume task_store contains the solution (the derivative)
        solution = task_plan.get('solution', task_plan['derivative_str'])

        # Handle multiple choice options
        options = {}
        if self.multiple_choice:
            correct_option = solution
            options["correct"] = correct_option
            options["incorrect option 1"] = "0"
            options["incorrect option 2"] = "1x"
            options["incorrect option 3"] = "2x^2" if "^2" in correct_option else "3x"

        return question, options, solution, self.metadata

class TrigonometricDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "function": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        functions = ["sin(x)", "cos(x)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for function in functions:
                    task_plan = {
                        "question_template": template,
                        "function": function
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        function = task_plan["function"]
        answer = "cos(x)" if function == "sin(x)" else "-sin(x)"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["1", 0, function]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(function)
        return question, options, answer, self.metadata


class TrigonometricIntegrationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "function": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        functions = ["sin(x)", "cos(x)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for function in functions:
                    task_plan = {
                        "question_template": template,
                        "function": function
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        function = task_plan["function"]
        answer = "-cos(x) + C" if function == "sin(x)" else "sin(x) + C"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["x + C", "0", "1"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(function)
        return question, options, answer, self.metadata


class PolynomialLimitGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "polynomial": "str",
        "limit_point": "int",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        polynomials = ["2x + 3", "x^2 - 5x + 6"]
        limit_points = [1, 2]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for poly in polynomials:
                    for point in limit_points:
                        task_plan = {
                            "question_template": template,
                            "polynomial": poly,
                            "limit_point": point
                        }
                        task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        poly = task_plan["polynomial"]
        point = task_plan["limit_point"]
        if poly == "2x + 3":
            answer = str(2 * point + 3)
        else:
            answer = str(point**2 - 5 * point + 6)

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["0", "infinity", "undefined"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(poly, point)
        return question, options, answer, self.metadata
    

class ChainRuleDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "expression": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        expressions = ["(x^2 + 1)^3", "(3x + 2)^2"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for expression in expressions:
                    task_plan = {
                        "question_template": template,
                        "expression": expression
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        expression = task_plan["expression"]
        answer = "3(x^2 + 1)^2 * 2x" if expression == "(x^2 + 1)^3" else "2(3x + 2) * 3"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["x^3", "2x", "x + 2"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(expression)
        return question, options, answer, self.metadata
    
class LogarithmicDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "expression": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        expressions = ["ln(x)", "log(x)", "ln(2x)", "log(3x^2)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for expression in expressions:
                    task_plan = {
                        "question_template": template,
                        "expression": expression
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        expression = task_plan["expression"]

        if expression == "ln(x)":
            answer = "1/x"
        elif expression == "log(x)":
            answer = "1/(x * ln(10))"
        elif expression == "ln(2x)":
            answer = "1/x"
        elif expression == "log(3x^2)":
            answer = "2/(x * ln(10))"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["0", "x", f"{expression} * x"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})

        question = template.format(expression)
        return question, options, answer, self.metadata

class ProductRuleDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "functions": "list",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        functions = [("x^2", "sin(x)"), ("x^3", "ln(x)"), ("e^x", "cos(x)")]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for func1, func2 in functions:
                    task_plan = {
                        "question_template": template,
                        "functions": [func1, func2]
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        func1, func2 = task_plan["functions"]
        if func1 == "x^2" and func2 == "sin(x)":
            answer = "2x*sin(x) + x^2*cos(x)"
        elif func1 == "x^3" and func2 == "ln(x)":
            answer = "3x^2*ln(x) + x^3/x"
        elif func1 == "e^x" and func2 == "cos(x)":
            answer = "e^x*cos(x) - e^x*sin(x)"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["0", "x*sin(x)", "e^x"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(func1, func2)
        return question, options, answer, self.metadata

class ExponentialDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "expression": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        expressions = ["e^x", "2^x", "e^(2x)", "3e^(x^2)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for expression in expressions:
                    task_plan = {
                        "question_template": template,
                        "expression": expression
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        expression = task_plan["expression"]

        # Determine the derivative based on the expression
        if expression == "e^x":
            answer = "e^x"
        elif expression == "2^x":
            answer = "ln(2) * 2^x"
        elif expression == "e^(2x)":
            answer = "2e^(2x)"
        elif expression == "3e^(x^2)":
            answer = "6xe^(x^2)"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["0", "x", f"2 * {expression}"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(expression)
        return question, options, answer, self.metadata

class SecondDerivativeGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "expression": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        expressions = ["x^3", "sin(x)", "cos(x)", "ln(x)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for expression in expressions:
                    task_plan = {
                        "question_template": template,
                        "expression": expression
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        expression = task_plan["expression"]

        if expression == "x^3":
            answer = "6x"
        elif expression == "sin(x)":
            answer = "-sin(x)"
        elif expression == "cos(x)":
            answer = "-cos(x)"
        elif expression == "ln(x)":
            answer = "-1/x^2"

        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["0", "sin(x)", "cos(x)"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})

        question = template.format(expression)
        return question, options, answer, self.metadata


class IntegrationByPartsGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "integrand": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        integrands = ["x * e^x", "x * ln(x)", "e^x * sin(x)"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for integrand in integrands:
                    task_plan = {
                        "question_template": template,
                        "integrand": integrand
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        integrand = task_plan["integrand"]
        
        if integrand == "x * e^x":
            answer = "x * e^x - e^x + C"
        elif integrand == "x * ln(x)":
            answer = "(x^2 / 2) * ln(x) - x^2 / 4 + C"
        elif integrand == "e^x * sin(x)":
            answer = "(e^x * sin(x) - e^x * cos(x)) / 2 + C"
        
        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["ln(x)", "1/x", "x^2"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(integrand)
        return question, options, answer, self.metadata

class ImplicitDifferentiationGenerator(CalcPlanGenerator):
    schema = {
        "question_template": "str",
        "equation": "str",
    }

    def __init__(self, metadata, multiple_choice=True, seed=42):
        self.metadata = metadata
        self.seed = seed
        self.multiple_choice = multiple_choice
        np.random.seed(seed)

    def enumerate_task_plans(self, task_store):
        equations = ["x^2 + y^2 = 1", "xy = 1", "x^2 + xy + y^2 = 1"]
        template_breakdown = self.metadata.templates_by_num_params
        for _, templates in template_breakdown.items():
            for template in templates:
                for equation in equations:
                    task_plan = {
                        "question_template": template,
                        "equation": equation
                    }
                    task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan["question_template"]
        equation = task_plan["equation"]
        
        if equation == "x^2 + y^2 = 1":
            answer = "-x/y"
        elif equation == "xy = 1":
            answer = "-y/x"
        elif equation == "x^2 + xy + y^2 = 1":
            answer = "(-2x - y) / (2y + x)"
        
        options = {"correct": answer}
        if self.multiple_choice:
            incorrect = ["1", "x/y", "y/x"]
            options.update({f"incorrect option {i+1}": val for i, val in enumerate(incorrect)})
        
        question = template.format(equation)
        return question, options, answer, self.metadata



template_path = 'math_annotations/calculus/implicit_diff_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = ImplicitDifferentiationGenerator(metadata, multiple_choice=True)
task_store = TaskStore(ImplicitDifferentiationGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')


for i in range(0, 10):  # Limit output to 10 tasks for simplicity
   random_task_plan = random.choice(all_tasks)
   task = generator._generate_task(random_task_plan)
   question, options, answer, _ = task
   print()
   print(question)
   print("Answer: ", options)