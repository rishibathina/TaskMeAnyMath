import os
import sys

from typing import Dict, List, Tuple

import numpy as np 
from sympy import symbols, Eq, solve, sqrt
import sympy as sp
from tqdm import tqdm
import math
import scipy.stats as stats

from baseMath import TaskGenerator
from task_storeMath import TaskStore
from utils import *
import json
import enum
from metadataMath import MathTemplateMetaData

class StatsProbTaskGenerator(TaskGenerator):
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
    
# Permutation and Combination Problems
# Conditional Probability
# Confidence Intervals
# Basic Probability Calculation
# Expected Value

class BasicProbabilityGenerator(StatsProbTaskGenerator):
    schema = {
        'question_template': 'str',
        'total_items': 'int'
    }
    
    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Convert the task plan into a string
        return json.dumps(task_plan)

    # Enumerate function
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params
        
        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for _ in range(self.num_tasks):
                    # Generate task plan for any template
                    task_plan = {
                        'question_template': template_text,
                        'total_items': np.random.randint(5, 50)  # Random total items (team size, dice sides, etc.)
                    }
                    task_store.add(task_plan)

    # Generate the task
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        total_items = task_plan['total_items']
        question = template.format(total_items)
        answer = str(1 / total_items)
        # Generate multiple-choice options
        if self.multiple_choice:
            option_1 = float(answer) + 0.01
            if option_1 > 1 or option_1 < 0:
                option_1 = 1
            option_2 = float(answer) - 0.01
            if option_2 > 1 or option_2 < 0:
                option_2 = 0
            option_3 = float(answer) - 0.05
            if option_3 > 1 or option_3 < 0:
                option_3 = 0.5
            options = {
                "incorrect option 1": str(option_1),
                "incorrect option 2": str(option_2),
                "incorrect option 3": str(option_3)
            }
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata
    

class PermutationCombinationGenerator(StatsProbTaskGenerator):
    schema = {
        'question_template': 'str',
        'n': 'int',
        'r': 'int'
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Convert the task plan into a string
        return json.dumps(task_plan)

    # Enumerate function
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params

        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for _ in range(self.num_tasks):
                    # Random values for n (total items) and r (chosen items)
                    n = np.random.randint(5, 20)
                    r = np.random.randint(1, n)
                    task_plan = {
                        'question_template': template_text,
                        'n': n,
                        'r': r
                    }
                    task_store.add(task_plan)

    # Generate the task
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        n = task_plan['n']
        r = task_plan['r']

        question = template.format(n=n, r=r)
        if "arranged" in template:  # This indicates a permutation problem
            answer = str(math.perm(n, r))
        elif "chosen" in template or "selected" in template:  # This indicates a combination problem
            answer = str(math.comb(n, r))
        
        # Generate multiple-choice options
        if self.multiple_choice:
            incorrect_option_1 = int(answer) + np.random.randint(1, 10)
            incorrect_option_2 = int(answer) - np.random.randint(1, 10)
            incorrect_option_3 = int(answer) + np.random.randint(10, 20)
            if incorrect_option_1 > 1 or incorrect_option_1 < 0:
                incorrect_option_1 = 1
            if incorrect_option_2 > 1 or incorrect_option_2 < 0:
                incorrect_option_2 = 0
            if incorrect_option_3 > 1 or incorrect_option_3 < 0:
                if incorrect_option_1 != 1:
                    incorrect_option_3 = 1
                elif incorrect_option_1 != 0:
                    incorrect_option_3 = 0
                else:
                    incorrect_option_3 = 0.5
            options = {
                "incorrect option 1": str(incorrect_option_1),
                "incorrect option 2": str(incorrect_option_2),
                "incorrect option 3": str(incorrect_option_3)
            }
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata
    
class ConditionalProbabilityGenerator(StatsProbTaskGenerator):
    schema = {
        'question_template': 'str',
        'prob_b': 'str',
        'prob_a_and_b': 'str'
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Convert the task plan into a string
        return json.dumps(task_plan)

    # Enumerate function
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params

        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for _ in range(self.num_tasks):
                    # Generate task plans with random probabilities
                    prob_b = np.round(np.random.uniform(0.2, 0.9), 2)
                    prob_a_and_b = np.round(np.random.uniform(0.01, prob_b), 2)
                    task_plan = {
                        'question_template': template_text,
                        'prob_b': prob_b,
                        'prob_a_and_b': prob_a_and_b
                    }
                    task_store.add(task_plan)

    # Generate the task
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        prob_b = float(task_plan['prob_b'])
        prob_a_and_b = float(task_plan['prob_a_and_b'])

        # Generate the question using the template
        question = template.format(prob_b=prob_b, prob_a_and_b=prob_a_and_b)
        
        # Calculate the answer using conditional probability formula
        answer = str(np.round(prob_a_and_b / prob_b, 2))

        # Generate multiple-choice options
        if self.multiple_choice:
            incorrect_option_1 = np.round(float(answer) + 0.1, 2)
            incorrect_option_2 = np.round(float(answer) - 0.1, 2)
            incorrect_option_3 = np.round(float(answer) + 0.05, 2)
            if incorrect_option_1 > 1 or incorrect_option_1 < 0:
                incorrect_option_1 = 1
            if incorrect_option_2 > 1 or incorrect_option_2 < 0:
                incorrect_option_2 = 0
            if incorrect_option_3 > 1 or incorrect_option_3 < 0:
                if incorrect_option_1 != 1:
                    incorrect_option_3 = 1
                elif incorrect_option_1 != 0:
                    incorrect_option_3 = 0
                else:
                    incorrect_option_3 = 0.5
            options = {
                "incorrect option 1": str(incorrect_option_1),
                "incorrect option 2": str(incorrect_option_2),
                "incorrect option 3": str(incorrect_option_3)
            }
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata

    
class ConfidenceIntervalGenerator(StatsProbTaskGenerator):
    schema = {
        'question_template': 'str',
        'n': 'int',
        'mean': 'str',
        'error_measure': 'str',
        'confidence_level': 'str',
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Convert the task plan into a string
        return json.dumps(task_plan)

    # Enumerate function (no if-else)
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params

        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for _ in range(self.num_tasks):
                    # Random values for n, mean, and error_measure
                    n = np.random.randint(30, 200)
                    mean = np.round(np.random.uniform(50, 100), 2)
                    error_measure = np.round(np.random.uniform(1, 10), 2)
                    confidence_level = np.random.choice([90, 95, 99])

                    # Create task plan (unified structure)
                    task_plan = {
                        'question_template': template_text,
                        'n': n,
                        'mean': mean,
                        'error_measure': error_measure,
                        'confidence_level': confidence_level,
                    }
                    task_store.add(task_plan)

    # Generate the task (also without if-else)
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        n = task_plan['n']
        mean = float(task_plan['mean'])
        error_measure = float(task_plan['error_measure'])
        confidence_level = float(task_plan['confidence_level'])

        # Get the corresponding Z-score for the confidence level
        z_score = stats.norm.ppf(1 - (1 - confidence_level / 100) / 2)

        # Calculate the margin of error
        margin_of_error = z_score * (error_measure / np.sqrt(n))
        lower_bound = np.round(mean - margin_of_error, 2)
        upper_bound = np.round(mean + margin_of_error, 2)

        # Generate the question using the unified template
        question = template.format(n=n, mean=mean, error_measure=error_measure, confidence_level=confidence_level)

        # Return the confidence interval as the answer
        answer = f"({lower_bound}, {upper_bound})"

        # Generate multiple-choice options
        if self.multiple_choice:
            incorrect_option_1 = f"({np.round(lower_bound + 1, 2)}, {np.round(upper_bound + 1, 2)})"
            incorrect_option_2 = f"({np.round(lower_bound - 1, 2)}, {np.round(upper_bound - 1, 2)})"
            incorrect_option_3 = f"({np.round(lower_bound + 0.5, 2)}, {np.round(upper_bound + 0.5, 2)})"
            options = {
                "incorrect option 1": incorrect_option_1,
                "incorrect option 2": incorrect_option_2,
                "incorrect option 3": incorrect_option_3
            }
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = answer
        else:
            deduped_options = {}

        return question, deduped_options, answer, self.metadata


class ExpectedValueGenerator(StatsProbTaskGenerator):
    schema = {
        'question_template': 'str',
        'value1': 'str',
        'value2': 'str',
        'prob1': 'str',
        'prob2': 'str',
    }

    def __init__(self, metadata, seed=42, num_tasks=100, multiple_choice=False):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.seed = seed
        self.multiple_choice = multiple_choice

    def _task_plan_to_str(self, task_plan):
        # Convert the task plan into a string
        return json.dumps(task_plan)

    # Enumerate function to create random expected value problems
    def enumerate_task_plans(self, task_store):
        np.random.seed(self.seed)
        templates_by_num_params = self.metadata.templates_by_num_params

        for num_params, templates in templates_by_num_params.items():
            for template_text in templates:
                for _ in range(self.num_tasks):
                    value1 = np.round(np.random.uniform(10, 100), 2)
                    value2 = np.round(np.random.uniform(10, 100), 2)
                    prob1 = np.round(np.random.uniform(0.1, 0.9), 2)
                    prob2 = np.round(1 - prob1, 2)

                    # Create the task plan
                    task_plan = {
                        'question_template': template_text,
                        'value1': value1,
                        'value2': value2,
                        'prob1': prob1,
                        'prob2': prob2,
                    }
                    task_store.add(task_plan)

    # Generate the task by calculating the expected value
    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        template = task_plan['question_template']
        value1 = float(task_plan['value1'])
        value2 = float(task_plan['value2'])
        prob1 = float(task_plan['prob1'])
        prob2 = float(task_plan['prob2'])
        expected_value = value1 * prob1 + value2 * prob2
        question = template.format(value1=value1, value2=value2, prob1=prob1, prob2=prob2)
        if self.multiple_choice:
            incorrect_option_1 = np.round(expected_value + np.random.uniform(1, 10), 2)
            incorrect_option_2 = np.round(expected_value - np.random.uniform(1, 10), 2)
            incorrect_option_3 = np.round(expected_value + np.random.uniform(5, 15), 2)
            options = {
                "incorrect option 1": str(incorrect_option_1),
                "incorrect option 2": str(incorrect_option_2),
                "incorrect option 3": str(incorrect_option_3)
            }
            deduped_options = {k: v for i, (k, v) in enumerate(options.items()) if v not in list(options.values())[:i]}
            deduped_options["correct"] = str(np.round(expected_value, 2))
        else:
            deduped_options = {}

        return question, deduped_options, str(np.round(expected_value, 2)), self.metadata

