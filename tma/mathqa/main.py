import random
import torch
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from geometry_task import (
    MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator,
    AngleGenerator, PerpendicularGenerator, SideLengthGenerator, ArcLengthGenerator,
    CircleGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator, PointSlopeGenerator,
    RemainderTheoremGenerator, QuadraticFormulaGenerator, TaskStore
)
from mathTrainingEnv import mathTrainingEnv
from trainRL import train_rl_agent_with_feedback

def main():
    Perimeter_template_path = r"C:\path\to\perimeter_templates.json"

    reasonings = {
        "WrongOption1": "Remember, the perimeter of a triangle is the sum of all its sides.",
        "WrongOption2": "For non-equilateral triangles, ensure all three sides are accounted for.",
        "WrongOption3": "Check your calculations and verify all sides are included."
    }

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    env = mathTrainingEnv(
        generator_class=PerimeterGenerator,
        template_path=Perimeter_template_path,
        tokenizer=tokenizer,
        multiple_choice=True
    )
    env.reasonings = reasonings  

    train_rl_agent_with_feedback(env, model, episodes=100)

if __name__ == "__main__":
    main()
