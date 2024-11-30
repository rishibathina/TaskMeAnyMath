import random
from geometry_task import (
    MathTemplateMetaData, PerimeterGenerator, MidpointGenerator,IntersectionGenerator, 
    PerimeterGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator,
     ArcLengthGenerator, CircleGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator,
    PointSlopeGenerator, RemainderTheoremGenerator, QuadraticFormulaGenerator, TaskStore
)
import random
import torch
from transformers import AutoTokenizer

class mathTrainingEnv:
    def __init__(self, generator_class, template_path, tokenizer, multiple_choice=True):
        """
        Initialize the RL environment with reasoning feedback for wrong answers.
        """
        metadata = MathTemplateMetaData(template_path=template_path)
        self.generator = generator_class(metadata=metadata, multiple_choice=multiple_choice)
        self.task_store = TaskStore(schema=generator_class.schema)
        self.generator.enumerate_task_plans(self.task_store)
        self.tasks = list(self.task_store)
        self.tokenizer = tokenizer
        self.multiple_choice = multiple_choice
        self.current_task = None
        self.reasonings = {}  

    def reset(self):
        """
        Reset the environment with a new task.
        """
        if not self.tasks:
            raise ValueError("No tasks available.")
        random_task_plan = random.choice(self.tasks)
        self.current_task = self.generator.generate(random_task_plan)
        return self._encode_state(self.current_task)

    def step(self, action):
        """
        Take a step in the environment, providing reasoning feedback for wrong answers.
        Args:
            action (int): The index of the selected option.

        Returns:
            state: Encoded state of the new task.
            reward: Reward for the chosen action.
            done: Whether the episode is complete.
        """
        question, options, correct_answer, _ = self.current_task

        if options[action] == correct_answer:
            reward = 1
            done = True 
            feedback = None
        else:
            reward = -0.5  
            done = False
            feedback = self.reasonings.get(options[action], "No specific feedback available.")

        if not done:
            random_task_plan = random.choice(self.tasks)
            self.current_task = self.generator.generate(random_task_plan)

        state = self._encode_state(self.current_task, feedback=feedback)
        return state, reward, done

    def _encode_state(self, task, feedback=None):
        """
        Encode the current task as a tokenized input for Hugging Face models.
        """
        question, options, _, _ = task
        input_texts = [f"Q: {question} Option: {opt}" for opt in options]
        if feedback:
            input_texts = [f"{text} Feedback: {feedback}" for text in input_texts]
        return self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
