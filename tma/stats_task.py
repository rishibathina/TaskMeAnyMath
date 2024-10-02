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
    
class BasicProbabilityGenerator(GeoPlanGenerator):
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
    

class PermutationCombinationGenerator(GeoPlanGenerator):
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
    
class ConditionalProbabilityGenerator(GeoPlanGenerator):
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

    
class ConfidenceIntervalGenerator(GeoPlanGenerator):
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


class ExpectedValueGenerator(GeoPlanGenerator):
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
