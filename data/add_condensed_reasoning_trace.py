import json
import os
from pathlib import Path
from time import sleep

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(Path(__file__).parents[1] / ".env")


# Define a Pydantic model for the condensed trace output
class CondensedTrace(BaseModel):
    condensed_reasoning_trace: str


base_prompt = """
You are an AI assistant specialized in processing and structuring complex reasoning data for machine learning applications. Your primary task is to compress and formalize a given verbose, human-readable reasoning trace into a structured, labeled sequence according to a specific methodology I will provide.

**Purpose:** This conversion is part of a project to train a language model to reason more efficiently by training on structured, condensed reasoning steps. The goal is to capture the logical flow of problem-solving, including the exploration of different ideas, points of confusion, blockages, and eventual breakthroughs. For this purpose, strict adherence to the specified format and labels is paramount, prioritizing structural conciseness and accurate labeling of reasoning steps (like exploration and blockage) over producing a flowing, grammatically perfect sentence for each step.

**Methodology:** I will provide the "Unified Trace Methodology". This includes the definition of each single-letter label and an example reasoning trace converted using this methodology. You **must** strictly follow this methodology for the subsequent trace you are asked to convert. Pay particular attention to using the designated labels for different types of reasoning steps (e.g., defining questions, derivations, drawing conclusions, identifying external knowledge).

**Key Requirements for Conversion:**
- Each step in the output trace must use *exactly one* of the defined labels followed by a concise, bracketed description.
- Descriptions should be brief, using keywords, symbols, or very short phrases sufficient to identify the core action or thought.
- Use the 'Z' label specifically for points where the reasoning expresses confusion, asks an internal question, or notes uncertainty.
- Use the 'B' label specifically for points where the reasoning identifies a blockage, hits a dead end, or abandons a specific approach.
- The sequence of labeled steps should reflect the order of thoughts presented in the original verbose trace, capturing the flow, including branching exploration paths and returns to core problems or new ideas after blockages.

---
Unified Approach Methodology:
We will use a sequential format where each line represents a step in the reasoning process, marked by a single-letter label followed by a concise, bracketed description.
Structure of Each Line:
LABEL [Concise Description]
LABEL: A single uppercase letter representing the type of reasoning step.
[Concise Description]: A brief summary using keywords, short phrases, symbols, or references to previous steps. The goal is conciseness, not human readability or full grammatical correctness. Use '...' for repeated or contextually obvious elements.
Vocabulary of Labels:
Here is a proposed set of labels and their meanings, aiming for consistency and covering the typical stages of problem-solving reasoning, including exploration and errors:
P: Problem Analysis / Restatement: Initial understanding of the problem, breaking it down, restating conditions.
Q: Question / Goal: A specific question being asked, a sub-goal identified, or the ultimate goal of the process. Can express confusion ("Z") implicitly or explicitly if needed, but Q is for defining the target.
I: Idea / Approach Initiation: Starting a new line of reasoning, proposing an approach, or exploring a specific concept. Often follows a 'B' or 'Z'.
S: Statement / Step: A general logical step, observation, or assertion made during the reasoning.
D: Derivation / Calculation: Performing a calculation, algebraic manipulation, or logical deduction from previous steps.
A: Application: Applying a definition, theorem, formula, or known property.
C: Conclusion (Intermediate): Drawing a conclusion from a series of steps within a specific line of reasoning. This is typically not the final answer.
E: External Knowledge / Property: Recalling or referencing a known mathematical property, theorem, definition, or fact about the problem domain or space (H, properties of numbers, etc.).
Z: Confusion / Question (Internal): Expressing internal confusion, asking a question to oneself, or noting uncertainty about a step or approach. This specifically captures exploration/doubt.
B: Blockage / Barrier: Identifying a point where the current approach is stuck, a necessary piece is missing, or a contradiction is found. This specifically captures a dead end or difficulty.
V: Verification / Check: Checking the correctness of a previous step, calculation, or assumption.
K: Key Result / Final Answer: The final result or the critical intermediate breakthrough that directly leads to the solution.

How to Capture Exploration/Dead Ends:
The sequential nature, combined with the specific labels, captures exploration:
An I starts a new approach.
Steps (S, D, A, C) follow, developing the idea.
A Z indicates confusion about the current path or a need for clarification. It might lead to a new I or a B.
A B indicates the current path has hit a wall or is abandoned. It is typically followed by a new I (a different approach) or a return to a prior Q or P.
The sequence itself shows the flow: I -> S -> D -> B -> I -> S -> A -> C -> K demonstrates trying one thing, getting blocked, trying another, succeeding.
---
Example Trace using the Unified Methodology
P [Understand goal: find y s.t. scaled x-y is OS]
Q [OS conditions: ||x-y||=d/√2 & <x-y, x'-y>=0]
S [{x-y} must be orthogonal with norm d/√2]
I [Check relationship between distance and orthogonality]
A [Apply Parallelogram Law]
D [||x-x'||² = ||x-y||² + ||x'-y||² - 2<x-y, x'-y>]
A [Substitute ||x-x'||=d, ||x-y||=||x'-y||=d/√2]
D [$d^2 = d^2/2 + d^2/2 - 2<x-y, x'-y> \implies <x-y, x'-y>=0$]
C [Orthogonality is implied if ||x-y||=d/√2 for all x∈S]
Q [Problem reduces to finding y s.t. ||x-y||=d/√2 ∀x∈S]
I [Attempt 1: Algebraic system from ||x-y||²=d²/2]
D [$2<x,y> = ||x||^2 + ||y||^2 - d^2/2$]
S [System of equations for y]
D [Subtract equations for x, x': $2<x-x', y> = ||x||^2 - ||x'||^2$]
E [From ||x-x'||=d: $<x,x'> = (||x||^2 + ||x'||^2 - d^2)/2$]
S [Algebraic system is consistent with initial problem info]
B [Blockage: Hard to solve this system for y in general H]
I [Attempt 2: Geometric view - Intersection of spheres]
I [y ∈ $\cap_{x \in S}$ sphere(x, d/√2)]
E [H is complete, spheres are closed]
Z [Need intersection to be non-empty? FIP?]
E [FIP: $\cap$ closed sets non-empty in complete space if finite $\cap$ is non-empty]
Q [Need to show finite $\cap_{i=1}^n$ sphere(x_i, d/√2) is non-empty for any ${x_i} \subset S$]
A [Pairwise intersection holds: $d\sqrt{2} > d$]
Z [Does pairwise intersection imply finite intersection for this structure?]
I [Key Insight: Structure of S in infinite H]
J [Property: Equidistant sets with dist d in infinite H can be mapped to scaled OS {y + (d/√2)e_x}]
A [Applying property: The given S *has* this structure relative to some y]
A [This structure ensures finite $\cap$ of spheres are non-empty]
E [H complete, spheres closed]
X [By FIP (finite $\cap$ non-empty) & H complete, infinite $\cap$ is non-empty]
K [y exists in the intersection]
C [Existence of y with ||x-y||=d/√2 is proven]
K [Conclusion: y exists such that {√2/d(x-y)} is an OS]
---

**Trace to Convert:**
"""


def get_condensed_trace(verbose_trace, client, retries=3, delay=1):
    """
    Get condensed reasoning trace from the model with retry logic and structured output

    Args:
        verbose_trace (str): The verbose reasoning trace to condense
        model: The Gemini model instance
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds

    Returns:
        str: The condensed trace, or None if all retries fail
    """
    prompt = base_prompt + str(verbose_trace)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CondensedTrace,
                    thinking_config=types.ThinkingConfig(thinking_budget=8000),
                    temperature=0.1,
                    max_output_tokens=60000,
                ),
            )

            # Extract the condensed trace from the structured response
            try:
                response_text = response.text
                response_dict = json.loads(response_text)
                condensed_reasoning_trace = response_dict.get("condensed_reasoning_trace")
                if condensed_reasoning_trace:
                    return condensed_reasoning_trace
            except (AttributeError, IndexError, KeyError) as e:
                print(f"Failed to parse structured response: {e}")
                return None

        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            sleep(delay)


def process_dataset(input_path, client, checkpoint_interval=10):
    """
    Process the dataset to add condensed reasoning traces

    Args:
        input_path (str): Path to input parquet file
        model: The Gemini model instance
        checkpoint_interval (int): How often to save checkpoints
    """
    checkpoint_path = "s1.1-cr/data/condensed_traces_checkpoint.parquet"

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        print("Found existing checkpoint, checking if we can resume...")
        checkpoint_df = pd.read_parquet(checkpoint_path)
        if not checkpoint_df["deepseek_thinking_trajectory_condensed"].isna().all():
            print("Resuming from checkpoint...")
            df = checkpoint_df
        else:
            print("Checkpoint exists but has no condensed traces, starting fresh...")
            df = pd.read_parquet(input_path)
    else:
        print("Loading dataset...")
        df = pd.read_parquet(input_path)

    # Initialize the new column if it doesn't exist
    if "deepseek_thinking_trajectory_condensed" not in df.columns:
        df["deepseek_thinking_trajectory_condensed"] = None
    total_rows = len(df)

    print(f"Processing {total_rows} rows...")
    progress_bar = tqdm(range(total_rows), desc="Processing rows")
    for idx in progress_bar:
        verbose_trace = df.at[idx, "deepseek_thinking_trajectory"]
        if pd.isna(verbose_trace):
            continue

        condensed_trace = get_condensed_trace(verbose_trace, client)
        df.at[idx, "deepseek_thinking_trajectory_condensed"] = condensed_trace

        # Save checkpoint if needed
        if (idx + 1) % checkpoint_interval == 0:
            checkpoint_path = "s1.1-cr/data/condensed_traces_checkpoint.parquet"
            df.to_parquet(checkpoint_path, index=False)
            progress_bar.set_postfix({"checkpoint": f"saved at {idx + 1} rows"})

    # Save final results
    output_path = "s1.1-cr/data/updated_data_with_condensed_traces.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nProcessing complete. Results saved to {output_path}")


if __name__ == "__main__":
    # Configure the Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")

    # Initialize the model
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Process the dataset
    input_path = "hf://datasets/simplescaling/s1K-1.1/data/train-00000-of-00001.parquet"
    df = process_dataset(input_path, client)
