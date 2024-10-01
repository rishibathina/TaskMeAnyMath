from typing import Dict, List, Tuple
import json
import numpy as np
from sympy import symbols, sqrt, exp, factor, expand
import scipy.stats

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

        if isinstance(question, str):
            question = question.replace("_", " ")

        if isinstance(answer, str):
            answer = answer.replace("_", " ")
        elif isinstance(answer, dict):  
            answer = {k: v.replace("_", " ") if isinstance(v, str) else v for k, v in answer.items()}

        task = {
            "question": question.replace("_", " "),
            "options": options if self.multiple_choice else None,
            "answer": answer,
            "task_plan": self._task_plan_to_str(task_plan),
            "math_metadata": math_metadata,
        }
        return task

class MeanGenerator(GeoPlanGenerator):
    RANGE = (1, 10)
    LIST_LENGTH = (5, 10)

    schema = {
        'question_template': 'str',
        'numbers': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        templates = self.metadata.templates

        for template in templates:
            for _ in range(100):
                numbers = np.random.randint(
                    self.RANGE[0],
                    self.RANGE[1],
                    np.random.randint(self.LIST_LENGTH[0], self.LIST_LENGTH[1])
                )
                task_plan = {
                    'question_template': template,
                    'numbers': str(list(numbers))
                }
                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, float], float, Dict]:
        template = task_plan['question_template']
        numbers = eval(task_plan['numbers'])

        mean_value = np.mean(numbers)

        question = template.format(numbers=numbers)
        answer = mean_value

        options = {}
        if self.multiple_choice:
            options["correct"] = round(mean_value, 2)
            options["incorrect_option_1"] = round(mean_value + np.random.uniform(-3, 3), 2)
            options["incorrect_option_2"] = round(mean_value + np.random.uniform(-5, 5), 2)
            options["incorrect_option_3"] = round(mean_value + np.random.uniform(-7, 7), 2)

        return question, options, answer, self.metadata


class MedianGenerator(GeoPlanGenerator):
    RANGE = (1, 10)
    LIST_LENGTH = (5, 10)

    def __init__(self, metadata: MathTemplateMetaData, template_data, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice
        self.template = template_data['median_template']['text']

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            numbers = np.random.randint(
                self.RANGE[0],
                self.RANGE[1],
                np.random.randint(self.LIST_LENGTH[0], self.LIST_LENGTH[1])
            )
            task_plan = {
                'numbers': str(sorted(numbers))
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, float], float, Dict]:
        numbers = eval(task_plan['numbers'])
        median_value = np.median(numbers)

        question = self.template.format(numbers=numbers)
        answer = median_value

        options = {}
        if self.multiple_choice:
            options["correct"] = median_value
            options["incorrect_option_1"] = median_value + np.random.choice([-2, -1, 1, 2])
            options["incorrect_option_2"] = median_value + np.random.choice([-3, -2, 2, 3])
            options["incorrect_option_3"] = median_value + np.random.choice([-4, -3, 3, 4])

        return question, options, answer, self.metadata


class ModeGenerator(GeoPlanGenerator):
    RANGE = (1, 10)
    LIST_LENGTH = (5, 10)

    def __init__(self, metadata: MathTemplateMetaData, template_data, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice
        self.template = template_data['mode_template']['text']

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            numbers = np.random.randint(
                self.RANGE[0],
                self.RANGE[1],
                np.random.randint(self.LIST_LENGTH[0], self.LIST_LENGTH[1])
            )
            task_plan = {
                'numbers': str(list(numbers))
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, int], int, Dict]:
        numbers = eval(task_plan['numbers'])
        mode_result = scipy.stats.mode(numbers)
        mode_value = mode_result.mode[0]

        question = self.template.format(numbers=numbers)
        answer = mode_value

        options = {}
        if self.multiple_choice:
            options["correct"] = mode_value
            options["incorrect_option_1"] = mode_value + np.random.choice([-2, -1, 1, 2])
            options["incorrect_option_2"] = mode_value + np.random.choice([-3, -2, 2, 3])
            options["incorrect_option_3"] = mode_value + np.random.choice([-4, -3, 3, 4])

        return question, options, answer, self.metadata



class BayesTheoremGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'P_A': 'str',  # Prior probability P(A)
        'P_B_given_A': 'str',  # Likelihood P(B|A)
        'P_B': 'str'  # Marginal likelihood P(B)
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            P_A = np.random.uniform(0.01, 0.5)  
            P_B_given_A = np.random.uniform(0.6, 0.9) 
            P_B = P_B_given_A * P_A + np.random.uniform(0.01, 0.3)

            task_plan = {
                'question_template': (
                    "Given the following values:\n"
                    "P(A) = {P_A}\n"
                    "P(B|A) = {P_B_given_A}\n"
                    "P(B) = {P_B}\n\n"
                    "Calculate P(A|B)."
                ),
                'P_A': str(P_A),
                'P_B_given_A': str(P_B_given_A),
                'P_B': str(P_B)
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        P_A = float(task_plan['P_A'])
        P_B_given_A = float(task_plan['P_B_given_A'])
        P_B = float(task_plan['P_B'])

        P_A_given_B = (P_B_given_A * P_A) / P_B 

        question = task_plan['question_template'].format(
            P_A=task_plan['P_A'], 
            P_B_given_A=task_plan['P_B_given_A'], 
            P_B=task_plan['P_B']
        )
        answer = str(round(P_A_given_B,2))

        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(round(P_A_given_B * np.random.uniform(0.8, 1.2),2))
            options["incorrect option 2"] = str(round(P_A_given_B * np.random.uniform(0.6, 1.4),2))
            options["incorrect option 3"] = str(round(P_A_given_B * np.random.uniform(0.4, 1.6),2))

        return question, options, answer, self.metadata



class SampleStdDeviationGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'data': 'str'  
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            data = np.random.randint(1, 101, size=np.random.randint(5, 15))

            task_plan = {
                'question_template': (
                    "Given the data set {data}, calculate the standard deviation."
                ),
                'data': str(data.tolist()) 
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        data = np.array(eval(task_plan['data']))
        
        std_dev = np.std(data, ddof=1)

        question = task_plan['question_template'].format(data=task_plan['data'])

        answer = str(round(std_dev, 2))

        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(round(std_dev * np.random.uniform(0.8, 1.2), 2))
            options["incorrect option 2"] = str(round(std_dev * np.random.uniform(0.6, 1.4), 2))
            options["incorrect option 3"] = str(round(std_dev * np.random.uniform(0.4, 1.6), 2))

        return question, options, answer, self.metadata


class PopulationStdDeviationGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'data': 'str' 
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            data = np.random.randint(1, 101, size=np.random.randint(5, 15))

            task_plan = {
                'question_template': (
                    "Given the population data set {data}, calculate the population standard deviation."
                ),
                'data': str(data.tolist())  
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        data = np.array(eval(task_plan['data']))

        std_dev = np.std(data, ddof=0)

        question = task_plan['question_template'].format(data=task_plan['data'])

        answer = str(round(std_dev, 2))

        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(round(std_dev * np.random.uniform(0.8, 1.2), 2))
            options["incorrect option 2"] = str(round(std_dev * np.random.uniform(0.6, 1.4), 2))
            options["incorrect option 3"] = str(round(std_dev * np.random.uniform(0.4, 1.6), 2))

        return question, options, answer, self.metadata


class CrossEntropyGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'true_probs': 'str',  
        'predicted_probs': 'str' 
    }

    def __init__(self, metadata: MathTemplateMetaData, multiple_choice=True, seed=42):
        super().__init__(metadata, seed=seed)
        self.multiple_choice = multiple_choice

    def enumerate_task_plans(self, task_store: TaskStore):
        for _ in range(100):
            true_probs = np.random.dirichlet(np.ones(5), size=1)[0]
            predicted_probs = np.random.dirichlet(np.ones(5), size=1)[0]

            true_probs = np.round(true_probs, 2)
            predicted_probs = np.round(predicted_probs, 2)

            true_probs[-1] = round(1 - np.sum(true_probs[:-1]),2)
            predicted_probs[-1] = round(1 - np.sum(predicted_probs[:-1]),2)

            task_plan = {
                'question_template': (
                    "Given the true probability distribution {true_probs} and predicted distribution {predicted_probs}, "
                    "calculate the cross-entropy."
                ),
                'true_probs': str(true_probs.tolist()),
                'predicted_probs': str(predicted_probs.tolist())
            }
            task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, Dict[str, str], str, Dict]:
        true_probs = np.array(eval(task_plan['true_probs']))
        predicted_probs = np.array(eval(task_plan['predicted_probs']))

        cross_entropy = -np.sum(true_probs * np.log(predicted_probs))

        question = task_plan['question_template'].format(
            true_probs=task_plan['true_probs'], 
            predicted_probs=task_plan['predicted_probs']
        )

        answer = str(round(cross_entropy, 2))

        options = {}
        if self.multiple_choice:
            options["correct"] = answer
            options["incorrect option 1"] = str(round(cross_entropy * np.random.uniform(0.8, 1.2), 2))
            options["incorrect option 2"] = str(round(cross_entropy * np.random.uniform(0.6, 1.4), 2))
            options["incorrect option 3"] = str(round(cross_entropy * np.random.uniform(0.4, 1.6), 2))

        return question, options, answer, self.metadata