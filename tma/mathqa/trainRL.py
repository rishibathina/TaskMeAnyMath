import random
from geometry_task import (
    MathTemplateMetaData, PerimeterGenerator, MidpointGenerator,IntersectionGenerator, 
    PerimeterGenerator, AngleGenerator, PerpendicularGenerator, SideLengthGenerator,
     ArcLengthGenerator, CircleGenerator, PointDistanceGenerator, PythagoreanTheoremGenerator,
    PointSlopeGenerator, RemainderTheoremGenerator, QuadraticFormulaGenerator, TaskStore
)
import random
import torch
import torch.optim as optim  
from transformers import AutoModelForSequenceClassification 


def train_rl_agent_with_feedback(env, model, episodes=100, lr=5e-5, gamma=0.99, epsilon=0.1):
    """
    Train the RL agent with reasoning feedback for wrong answers.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, len(env.current_task['options']) - 1)  # Explore
            else:
                with torch.no_grad():
                    logits = model(**state).logits
                    action = torch.argmax(logits).item()  # Exploit

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Compute loss and update model
            optimizer.zero_grad()
            logits = model(**state).logits
            loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([action]))
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    return total_rewards
