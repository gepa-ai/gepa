# ⭐ AnyMaths: GEPA Adapter for Solving Math Word Problems ⭐
**AnyMaths Adapter** is a GEPA Adapter for any dataset that contains math word problems of varying complexity and structure. It is designed to handle a wide range of mathematical tasks, including arithmetic, algebra, reasoning, and more.

---

### 📢 Auxiliary requirements
Some auxiliary requirements for using the AnyMaths Adapter are found in `src/gepa/adapters/anymaths_adapter/requirements.txt`. Simply install the required packages listed in that file via:
* If using `uv`:
    ```bash
    uv pip install -r src/gepa/adapters/anymaths_adapter/requirements.txt
    ```
* If using `pip`:
    ```bash
    pip install -r src/gepa/adapters/anymaths_adapter/requirements.txt
    ```

### ✍️ Preparing the dataset
In `src/gepa/examples/anymaths-bench/train_anymaths.py`, a sample function to prepare a dataset is provided via `init_dataset`. This function demonstrates how to load and preprocess the dataset for training and evaluation. Notably, it includes steps for data augmentation and splitting the dataset into training, validation, and test sets. We recommend to find and download datasets from [Hugging Face dataset hub](https://huggingface.co/datasets).

#### Best format for a custom dataset
If you have a custom dataset, it is best to follow the following schema:
```
{
    "question": ...,
    "solution": ...,
    "answer": ...
}
```
Remarks:
- `question` must be a string/text.
- `solution` must be a string/text.
- `answer` must be a purely numerical, no other text or units associated.

We recommend you to upload your custom dataset to the Hugging Face dataset hub to fully utilize `datasets.load_dataset`.

---

### 🧰 Adapter Design
The AnyMaths Adapter can work for any LiteLLM supported providers (e.g., OpenAI, Google Vertex, HuggingFace, Groq, vLLM, Ollama, etc.). For this instance, we opt to choose Ollama to show that **this adapter can work for local use if one has no access to expensive GPUs or paid APIs.** But, you may freely choose this adapter with any other LiteLLM-supported provider.

---

### ✍️ Preparing the seed prompt
The seed prompt is the initial instruction you provide to the base (target) model. It sets the context for the task at hand and this prompt evolves or changes over time toward maximizing the model's performance. The default failure score (i.e., score if the model outputs are incorrect or does not satisfy a set metric) is zero.

Set the seed prompt in a separate directory under `src/gepa/examples/anymaths-bench/prompt-templates`. Inside this directory is a file `instruction_prompt.txt` which contains the seed prompt.

---

### 🪞 Specifying the reflection LM
The reflection LM is a language model used to generate feedback based on the output by the base model. You can specify the reflection LM by setting the `reflection_lm` argument when calling the `gepa.optimize` function.

A sample `reflection_lm` function call can be found in `src/gepa/examples/anymaths-bench/train_anymaths.py`.

---

### 🏃‍♂️ Running a full AnyMaths Adapter training
To run a full training session using the AnyMaths Adapter, you can use the following command:
```bash
python src/gepa/examples/anymaths-bench/train_anymaths.py --anymaths_dset_name ... --train_size ... --val_size ... --test_size ... --base_lm ... --use_api_base --api_base_url ... --reflection_lm ... --use_api_reflection --api_reflection_url ... --reflection_minibatch_size ... --budget ... --max_litellm_workers ... --seed ...
```
- `--anymaths_dset_name`: The Hugging Face `Dataset` name to use for training (e.g., `"openai/gsm8k"`, `"MathArena/aime_2025"`).
- `--train_size`: The size of the training set to use.
- `--val_size`: The size of the validation set to use.
- `--test_size`: The size of the test set to use.
- `--base_lm`: The base language model to use for GEPA training (e.g. `"ollama/qwen3:4b"`).
- `--use_api_base`: Enable this flag if you want to use the Ollama API for the base model. Otherwise, do not include this in your arguments if you are using provider APIs (e.g., OpenAI, Google Vertex, etc.).
- `--api_base_url`: (Base model) The URL to get completions for the base model. Example: Ollama uses the default `http://localhost:11434`. There is no need to set this up if you are using provider APIs. **Note: API keys and provider credentials must be set beforehand.**
- `--reflection_lm`: The reflection language model to generate feedback from base model outputs (e.g., `"ollama/qwen3:8b"`).
- `--use_api_reflection`: Similar with `--use_api_base`. Enable this flag if you want to use a specific endpoint to get completions from the reflection model.
- `--reflection_minibatch_size`: The minibatch size for the reflection LM to reflect against (default is 8).
- `--max_litellm_workers`: The maximum number of LiteLLM workers to use.
- `--budget`: The budget for the GEPA training (default is 500).
- `--seed`: The seed for the random number generator for reproducibility (default is 0).

#### 📓 Training command examples

1. (Purely) **Using Ollama**:
    ```bash
    python src/gepa/examples/anymaths-bench/train_anymaths.py --anymaths_dset_name "openai/gsm8k" --train_size 50 --val_size 50 --test_size 50 --base_lm "ollama/qwen3:4b" --use_api_base --api_base_url "http://localhost:11434" --reflection_lm "ollama/qwen3:8b" --use_api_reflection --api_reflection_url "http://localhost:11434" --reflection_minibatch_size 8 --budget 500 --max_litellm_workers 4 --seed 0
    ```
2. (Purely) **Using Google Vertex** for Gemini users:
    ```bash
    python src/gepa/examples/anymaths-bench/train_anymaths.py --anymaths_dset_name "openai/gsm8k" --train_size 50 --val_size 50 --test_size 50 --base_lm "vertex_ai/gemini-2.5-flash-lite" --reflection_lm "vertex_ai/gemini-2.5-flash" --reflection_minibatch_size 8 --budget 500 --max_litellm_workers 4 --seed 0
    ```
3. **Using Google Vertex as base (target) LM, Ollama as reflection LM**:
    ```bash
    python src/gepa/examples/anymaths-bench/train_anymaths.py --anymaths_dset_name "openai/gsm8k" --train_size 50 --val_size 50 --test_size 50 --base_lm "vertex_ai/gemini-2.5-flash-lite" --reflection_lm "ollama/qwen3:8b" --use_api_reflection --api_reflection_url "http://localhost:11434" --reflection_minibatch_size 8 --budget 500 --max_litellm_workers 4 --seed 0
    ```
4. **Using Ollama as base (target) LM, Google Vertex as reflection LM**:
    ```bash
    python src/gepa/examples/anymaths-bench/train_anymaths.py --anymaths_dset_name "openai/gsm8k" --train_size 50 --val_size 50 --test_size 50 --base_lm "ollama/qwen3:4b" --use_api_base --api_base_url "http://localhost:11434" --reflection_lm "vertex_ai/gemini-2.5-flash" --reflection_minibatch_size 8 --budget 500 --max_litellm_workers 4 --seed 0
    ```

Once the training has completed, you may replace the optimal prompt found in `src/gepa/examples/anymaths-bench/prompt-templates/optimal_prompt.txt`.

---

### 🔬 Model evaluation after GEPA training
`src/gepa/examples/anymaths-bench/eval_default.py` is used to perform model evaluation on the test split. Feel free to modify this script to fit your custom evaluation scheme. Example: `"openai/gsm8k"` - `test` is the dataset split used for benchmarking. The evaluation scores will be displayed in the terminal once the evaluation has been completed.

How to run the evaluation script:
1. **Using Ollama**:
    ```bash
    python src/gepa/examples/anymaths-bench/eval_default.py --anymaths_dset_name "openai/gsm8k" --model "ollama/qwen3:4b" --use_api_url --api_url "http://localhost:11434" --batch_size 8 --max_litellm_workers 4 --which_prompt "seed"
    ```
2. **Use Google Vertex** for Gemini users:
    ```bash
    python src/gepa/examples/anymaths-bench/eval_default.py --anymaths_dset_name "openai/gsm8k" --model "vertex_ai/gemini-2.5-flash-lite" --batch_size 8 --max_litellm_workers 4 --which_prompt "seed"
    ```
- `--anymaths_dset_name`: The name of the AnyMaths dataset to use for evaluation (default is `"openai/gsm8k"`).
- `--model`: The model to evaluate (default is `"ollama/qwen3:4b"`).
- `--use_api_url`: Whether to use the API URL (default is `False`).
- `--api_url`: The API URL to use (default is `"http://localhost:11434"`).
- `--batch_size`: The batch size for evaluation (default is `8`).
- `--max_litellm_workers`: The maximum number of LiteLLM workers to use (default is `4`).
- `--which_prompt`: The prompt to use for evaluation (default is `"seed"`, choices are `"seed"` and `"optimized"`).

**Note: The model that was used in GEPA training must also be the same model in performing model evaluation.**

---

### 🧪 Experiments
| Dataset | Base LM | Reflection LM | Accuracy, % (Before GEPA) $\uparrow$ | Accuracy, % (After GEPA) $\uparrow$ | GEPA Budget | Train-Val-Test Split Samples Used in GEPA Optimization |
| ------- | ------- | ------------- | ---------------------- | --------------------- | ------------ | ------ |
| `"openai/gsm8k"` | `"ollama/qwen3:4b"` | `"ollama/qwen3:8b"` | 18 | 23 (**+5**) | 500 | 50-50-50 |
| `"openai/gsm8k"` | `"vertex_ai/gemini-2.5-flash-lite"` | `"vertex_ai/gemini-2.5-flash"` | 31 | 33 (**+2**) | 500 | 50-50-50 |
| `"openai/gsm8k"` | `"ollama/qwen3:0.6b"` | `"ollama/qwen3:8b"` | 7 | 5 (**-2**) | 500 | 50-50-50 |
| `"openai/gsm8k"` | `"ollama/gemma3:1b"` | `"ollama/gemma3:4b"` | 9 | 38 (**+29**) | 500 | 50-50-50 |

**Notice of WIP**: More tests will be done soon on other models (preferrably, small language models first).

---

### 🏦 Prompt bank of optimal prompts

* Model: `"ollama/qwen3:4b"`, Dataset: `"openai/gsm8k"`, Budget: `500`:
    ```
    ### Task Instruction: Solve Multi-Step Mathematical Problems with Precision and Contextual Understanding

    You are tasked with solving problems that require careful parsing of contextual information, breaking down multi-step calculations, and ensuring accuracy in arithmetic and logical reasoning. Follow these steps to address diverse problem types (e.g., percentages, cost calculations, score determination, and distance computations):

    ---

    #### **1. Parse the Problem**
    - **Identify Key Values**: Extract numbers, percentages, fractions, and relationships (e.g., "40% of 60 students," "6 more than half of Ella\'s score").
    - **Understand Relationships**: Determine if values are additive, multiplicative, or comparative (e.g., "round trips" imply doubling one-way distances, "cost per item" requires multiplication).
    - **Clarify Ambiguities**: Resolve unclear phrasing (e.g., "half the score" refers to half the total items, not half the incorrect answers).

    ---

    #### **2. Break Down the Problem**
    - **Segment into Steps**: Divide the problem into smaller, manageable parts (e.g., calculate individual components before summing).
    - **Apply Formulas**: Use appropriate mathematical operations (e.g., percentage = part/whole × 100, total cost = (item count × price)).
    - **Account for Context**: Adjust calculations based on problem specifics (e.g., "round trip" requires doubling one-way distance, "score" may involve subtracting incorrect answers from total items).

    ---

    #### **3. Perform Calculations**
    - **Use Precise Arithmetic**:
        - For percentages: $ \\text{Percentage} \\times \\text{Total} $.
        - For fractions: $ \\frac{\\text{Numerator}}{\\text{Denominator}} \\times \\text{Value} $.
        - For multi-step operations: Follow order of operations (PEMDAS) and verify intermediate results.
    - **Avoid Common Errors**:
        - Misinterpreting phrases like "half the score" (e.g., half of total items, not half of incorrect answers).
        - Confusing "round trips" (up + down) with single trips.
        - Incorrectly applying percentages to the wrong base (e.g., 40% of students vs. 40% of total score).

    ---

    #### **4. Validate the Answer**
    - **Check Logical Consistency**: Ensure results align with problem constraints (e.g., total students = sum of groups, total cost = sum of individual costs).
    - **Verify Units and Formatting**: Confirm answers match required formats (e.g., boxed numbers, currency symbols, or percentage notation).
    - **Cross-Validate with Examples**: Compare calculations against similar problems (e.g., "If 40% of 60 students = 24, then 60 - 24 = 36").

    ---

    #### **5. Finalize the Response**
    - **Present the Answer Clearly**: Use the exact format requested (e.g., `\boxed{36}` for numerical answers, `$77.00` for currency).
    - **Include Step-by-Step Reasoning**: Explicitly show calculations (e.g., `18 * $2.50 = $45.00`, `8 * $4.00 = $32.00`).
    - **Highlight Key Decisions**: Note critical choices (e.g., "Half of Ella's score = 36 items / 2 = 18 items").

    ---

    ### **Examples of Problem Types**
    1. **Percentage Problems**:
        - *Input*: "40% of 60 students got below B."
        - *Solution*: $ 0.40 \\times 60 = 24 $, $ 60 - 24 = 36 $.
    2. **Cost Calculations**:
        - *Input*: "18 knobs at $2.50 each and 8 pulls at $4.00 each."
        - *Solution*: $ 18 \\times 2.50 + 8 \\times 4.00 = 45 + 32 = 77 $.
    3. **Score Determination**:
        - *Input*: "Ella got 4 incorrect answers; Marion got 6 more than half of Ella's score."
        - *Solution*: Total items = 40, Ella's correct = 36, half = 18, Marion = 18 + 6 = 24.
        
    ---

    ### **Key Niche Information**
    - **Percentages**: Always apply to the total (e.g., 40% of 60 students = 24 students, not 40% of 40 items).
    - **Round Trips**: Double one-way distances (e.g., 30,000 feet up + 30,000 feet down = 60,000 per trip).
    - **Score Calculations**: Subtract incorrect answers from total items (e.g., 40 items - 4 incorrect = 36 correct).
    - **Currency Formatting**: Use decimal points and symbols (e.g., `$77.00`, not `77`).

    ---

    ### **Final Output Format**
    Always conclude with:
        `Final Answer: \boxed{<result>}`
    For non-numeric answers, use:
        `Final Answer: <result>`

    Ensure calculations are explicitly shown and errors are corrected based on problem context.
    ```
* Model: `"vertex_ai/gemini-2.5-flash-lite"`, Dataset: `"openai/gsm8k"`, Budget: `500`:
    ```
    You are an AI assistant that solves mathematical word problems. You will be given a question and you need to provide a step-by-step solution to the problem. Finally, you will provide the answer to the question.

    When outputting the final answer, make sure there are no other text or explanations included, just the answer itself.

    The following fields are what you need to include in your response:
    - final_answer: The final answer to the question.
    - solution_pad: The step-by-step solution to the problem.

    Here are specific guidelines for generating your response:

    1.  **Understand the Problem Thoroughly:** Carefully read and analyze the word problem to ensure a complete understanding of all given information, constraints, and the specific question being asked. Pay close attention to units and how different quantities relate to each other.

    2.  **Formulate the Step-by-Step Solution (solution_pad):**
        *   Develop a clear, logical, and sequential step-by-step solution. Each step should be a distinct operation or deduction required to move closer to the final answer.
        *   Clearly state what is being calculated or determined in each step.
        *   Perform all necessary calculations with high precision and accuracy. Double-check all numerical operations (addition, subtraction, multiplication, division, etc.) to prevent errors.
        *   If the problem involves converting between different forms of a quantity (e.g., converting a monetary value into a count of items, or time units), explicitly show this conversion as a step.
            *   **Domain-Specific Interpretation Example:** If Barry has "$10.00 worth of dimes", first convert this value to the number of dimes (since a dime is $0.10, Barry has $10.00 / $0.10 = 100 dimes). If the problem then states Dan has "half that amount" and asks for the number of dimes Dan has, interpret "half that amount" as half the *number* of dimes Barry has (100 dimes / 2 = 50 dimes), rather than half the monetary value. Always aim for the most logical interpretation that leads to the requested unit in the final answer.
        *   The `solution_pad` field must *only* contain the clean, direct step-by-step solution. Do not include any internal monologues, self-corrections, re-evaluations, alternative thought processes, or debugging notes within this field.

    3.  **Calculate and Output the Final Answer:**
        *   Based on your thoroughly computed step-by-step solution, determine the exact numerical answer to the question.
        *   The `final_answer` field must contain *only* the numerical value. Do not include any currency symbols (e.g., "$"), units (e.g., "dimes", "hours"), or any other descriptive text or explanation in this field. For example, if the answer is 4625 dollars, output `4625`. If the answer is 52 dimes, output `52`.
        *   Ensure the final answer numerically matches the result of your `solution_pad` calculations.'
    ```
* Model: `"ollama/qwen3:0.6b"`, Dataset: `"openai/gsm8k"`, Budget: `500`:
    ```
    ### Instruction for Solving Math Word Problems

    **Task Description:**
    Solve multi-step math word problems by carefully analyzing the relationships between quantities, translating them into mathematical expressions, and performing accurate calculations. Ensure all components of the problem are addressed, and verify that percentages, fractions, and arithmetic operations are applied correctly.

    **Key Requirements:**
    1. **Parse Relationships:**
    - Identify explicit and implicit relationships (e.g., "20 fewer than," "1/10 less," "10 more than").
    - Define variables clearly (e.g., let Arts = x, then Maths = x - 20).

    2. **Translate to Equations:**
    - Convert word-based relationships into algebraic expressions or equations.
    - For percentage changes, apply the correct formula (e.g., "1/10 less" means 90% of the original value).

    3. **Account for All Components:**
    - Ensure all subjects, quantities, or data points mentioned in the problem are included in the final calculation.
    - Aggregate totals by summing individual values (e.g., total marks = sum of all subject scores).

    4. **Verify Arithmetic and Logic:**
    - Check for arithmetic errors (e.g., 1/10 of 70 = 7, not 70 - 7 = 63).
    - Validate that the solution aligns with the problem’s constraints (e.g., no negative scores unless explicitly allowed).

    5. **Document Step-by-Step Reasoning:**
    - Break down the problem into logical steps, explicitly showing calculations (e.g., total birds = sum of daily totals).
    - Use parentheses and order of operations to avoid errors (e.g., 5 sites × 7 birds/site = 35 birds).

    6. **Final Answer Validation:**
    - Ensure the final answer matches the problem’s question (e.g., total marks, average per site, or time difference).
    - Recheck all steps to confirm consistency with the problem’s context.

    **Example Application:**
    For a problem like:
    *"Amaya scored 20 fewer in Maths than Arts. She scored 10 more in Social Studies than Music. If she scored 70 in Music and 1/10 less in Maths, what is the total marks?"*
    - Define variables: Arts = x, Maths = x - 20, Social Studies = 70 + 10 = 80.
    - Calculate Maths: 1/10 less than Arts → Maths = x - 20 = 0.9x.
    - Solve for x: x - 20 = 0.9x → x = 200 (Arts), Maths = 180.
    - Total marks = 70 (Music) + 180 (Maths) + 200 (Arts) + 80 (Social Studies) = **296**.

    **Error Prevention:**
    - Avoid misinterpreting "1/10 less" as 1/10 of the value (instead, it means 90% of the original).
    - Ensure all subjects are included (e.g., Music, Maths, Arts, Social Studies in the example).
    - Double-check totals by summing individual components.
    ```
* Mode: `"ollama/gemma3:1b"`, Dataset: `"openai/gsm8k"`, Budget: `500`:
    ```
    You are a specialized assistant designed to solve complex, realistic word problems, prioritizing accuracy and detailed reasoning. Your primary goal is to meticulously determine the numerical answer while demonstrating a thorough understanding of the problem's context, constraints, and relevant domain knowledge. You will be provided with problem descriptions that often include specific units, quantities, and contextual details, aiming for scenarios mirroring real-world applications like logistics, resource management, and engineering calculations.

    **Here's your process:**

    1.  **Deconstruction & Contextual Understanding:** Carefully read the entire problem description. Identify *all* numerical values, units (e.g., kg, mph, minutes, dollars, meters, seconds), and relevant contextual details. Pay close attention to *all* constraints or limitations mentioned (e.g., “cannot exceed,” “up to,” “at most,” “within a tolerance of”).  Crucially, identify the *assumptions* embedded within the problem – what is being taken for granted?  For instance, is the problem implicitly stating that surfaces are flat, or that materials behave ideally?

    2.  **Unit Analysis and Conversion:**  Recognize that the problem fundamentally involves numerical calculations. However, rigorous unit conversion is paramount. Ensure 
    all calculations are performed with appropriate units. If units are mixed, perform conversions *accurately*. Specifically, you must be able to handle conversions between common units: kilograms (kg), miles per hour (mph), minutes, dollars, meters, seconds, Newtons (N), cubic meters (m³), and other quantities.  **Important:** Weight and mass are frequently confused.  Remember that weight is a force (typically measured in Newtons – N), while mass is a measure of matter (typically measured in kilograms – kg). Always consider the context when given a weight value – it *usually* implies the force due to gravity.

    3.  **Strategic Approach Selection:** For most problems, a straightforward algebraic approach will be effective.  Setting up equations based on the given information is usually the most efficient method.  However, consider scenarios where a simpler calculation (e.g., direct multiplication or division) is sufficient.  Furthermore, recognize that some problems may benefit from applying principles of physics (e.g., Newton’s Laws of Motion, conservation of energy) to derive equations.

    4.  **Step-by-Step Calculation with Intermediate Results:** Clearly articulate *each* step of your calculation, showing all intermediate results.  This is absolutely crucial for verification, debugging, and demonstrating your thought process. Include units with every calculation.

    5.  **Final Answer with Units and Precision:** Provide the final numerical answer, *always* including the correct units.  Pay attention to the level of precision required by the problem – often, rounding will be necessary.

    6.  **Critical Considerations – Domain-Specific Knowledge is Key:**

        *   **Weight & Mass:** *Master* the distinction between weight and mass.  Weight is force (N), mass is matter (kg).
        *   **Velocity & Distance:** Remember the relationship: distance = velocity * time. Be mindful of velocity as a vector (magnitude and direction).
        *   **Area & Volume:**  Be proficient with basic geometric formulas (rectangles, cubes, cylinders, spheres – relevant formulas will be provided in the problem).
        *   **Cargo Loading:**  Problems frequently involve loading crates, where the maximum weight capacity of a crate (typically 20 kg) is a critical constraint. Consider the impact of sub-optimal loading arrangements.
        *   **Fluid Mechanics (where applicable):**  Some problems may involve fluid flow, requiring knowledge of concepts like pressure, viscosity, and flow rate.
        *   **Energy & Work:** Be familiar with the concepts of work, potential energy, kinetic energy, and their relationships.
        *   **Linear Motion:** Understand concepts such as acceleration, displacement, and average speed.
        *   **Rotational Motion:** When problems involve rotating objects, understanding angular velocity, angular acceleration, torque, and moment of inertia are essential.

    7.  **Verification and Validation:**  After arriving at a solution, briefly outline *how* you verified your answer. Did you check your units?  Did you perform a sanity check (e.g., is the answer physically plausible)?

    8. **Error Handling:** If a problem is ambiguous or contains conflicting information, state your assumptions and explain how they affect your solution.
    ```

---

### 🔍 Observations on the structure of the derived optimal prompt
- For small language models:
    * Goal-oriented: The prompt starts by clearly stating the overall task, which is to solve multi-step math problems with precision.
    * Chain-of-Thought: It breaks down the problem-solving process into a detailed, numbered sequence of five steps: **Parse**, **Break Down**, **Calculate**, **Validate**, and **Finalize**.
    * Instruction Detail: Each step includes specific instructions on how to perform the task, such as identifying key values, applying formulas, and avoiding common errors.
    * Few-shot Learning: The prompt provides concrete **examples** of different problem types (percentage, costs, scores) to show the model how to apply the instructions.
    * Knowledge Base: It includes *key niche information* section that acts as a mini-rulebook, highlighting specific details and common pitfalls like "round trips" and currency formatting.
    * Structured Output: The prompt ends by defining a strict **final output format** to ensure the model's answer is consistent and easy to read.
    * Contextual Awareness: The prompt encourages the model to consider the broader context of the problem, including any implicit assumptions or constraints that may not be explicitly stated. Filed under *emergent* behavior.

- For provider models:
    * Fewer tokens: The prompt is more concise, using fewer tokens to convey the same information, which can lead to faster processing and lower costs.
    * Straightforward: Main instruction and output format are placed at the first parts of the prompt. Detailed guidelines are provided in a structured manner to facilitate understanding after the main instruction and output format.

### ❓ Weird results and observations
- In the case of `"ollama/qwen3:0.6b"`, the score did not improve as expected with additional context. Moreover, in the optimal prompt, the specific instruction to generate the expected JSON object was removed. In a similar fashion, we can also observe this omission in the optimal prompt for `"ollama/qwen3:4b"`.

---

### 👨‍🔬 Contributor
This adapter was contributed by **Emmanuel G. Maminta**. [[LinkedIn]](https://linkedin.com/in/egmaminta) [[GitHub]](https://github.com/egmaminta)