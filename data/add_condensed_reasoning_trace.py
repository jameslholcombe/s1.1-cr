reasoning_trace = "test"

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

prompt = base_prompt + reasoning_trace
