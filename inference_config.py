# --- Configuration for ACE-Agent ---
# IMPORTANT: Review and update these settings before running the agent.

# Ollama Server Configuration
# Set this to the URL of your running Ollama instance.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# File Paths (Defaults)
# These should be relative to the running script or provided via CLI
DEFAULT_INPUT_DIR = "INFERSET"
DEFAULT_OUTPUT_BASE_DIR = "EXPERIMENT_RESULTS"

# Database Configuration
DEFAULT_DB_NAME = "agent_memory.sqlite"

# OLLAMA Models
# Ensure these models are installed (pulled) in your Ollama instance before running.
#
# Reference models used in experiments:
# "llama3.3:70b", "llama3.1:8b", "mistral:7b-instruct", "mixtral:8x7b", "mixtral:8x22b"
#
# Note: "finetuned-mistral" was experimental and is not included in the release.
DEFAULT_MODELS = [
    "llama3.3:70b", "llama3.1:8b", "mistral:7b-instruct", "mixtral:8x7b", "mixtral:8x22b"
]



GENERATOR_PROMPT_TEMPLATE = """
System: You are an expert in German text simplification. Your goal is to simplify the given text to the target level.
Task:
1.  **Review the "Strategy Cheatsheet"** below for general heuristics and rules.
2.  **Analyze the Query:** Read the user's text and their target complexity level.
3.  **Formulate a Step-by-Step Strategy:** Based on the cheatsheet and the query, formulate a specific plan.
4.  **Execute the Simplification:** Apply your strategy to produce the final simplified text.
5.  **Return** both your *strategy* and the final *solution* in the required format.

**Strategy Cheatsheet (Current Memory M_i):**
```
{memory}
```
**User Query (x_i):**
"{query}"
**Target Level:**
"{target_level}"
"""

REFLECTOR_PROMPT_TEMPLATE = """
System: You are a critical analyst reviewing the work of a German simplification agent.

Task:
1.  **Review the Task:** An agent was given a query ("{query}").
2.  **Review the Agent's Work:**
    - Strategy used: "{strategy}"
    - Solution produced: "{solution}"
3.  **Review the Cheatsheet Used:** The agent used this cheatsheet:
    ```
    {old_memory}
    ```
4.  **Critique:** Analyze the agent's strategy and solution.
    - Was the simplification successful according to Leichte Sprache rules?
    - Was the strategy appropriate? Did it miss anything?
    - Did the solution preserve the original meaning?
5.  **Distill Actionable Insights:** Based on your critique, list concise, actionable insights or lessons learned that could improve the cheatsheet for future tasks.

Return ONLY the distilled insights/lessons learned. Do NOT rewrite the cheatsheet here.
"""

CURATOR_PROMPT_TEMPLATE = """
System: You are an editor responsible for maintaining a cheatsheet for a German simplification agent.

Task:
1.  **Review the Old Cheatsheet (M_i):**
    ```
    {old_memory}
    ```
2.  **Review the Reflection Insights:** An analyst reviewed the last task and provided these insights:
    ```
    {insights}
    ```
3.  **Integrate and Refine:** Rewrite the *entire* cheatsheet (M_i+1), incorporating the valuable insights from the reflection.
    - Add new generalizable rules or specific examples identified in the insights.
    - Refine or correct existing rules based on the insights.
    - Prune redundant or low-value entries.
    - Ensure the cheatsheet remains concise and well-organized.

Return ONLY the text for the New, Updated Cheatsheet (M_i+1).
"""

CURATOR_V1_PROMPT_TEMPLATE = """
System: You are a meta-analyst and editor. Your job is to improve a "cheatsheet" of strategies for a German simplification agent.
Task:
1.  **Review the Task:** An agent was given a query ("{query}").
2.  **Review the Agent's Work:**
    - Strategy used: "{strategy}"
    - Solution produced: "{solution}"
3.  **Review the Old Cheatsheet (M_i):** This is the set of strategies the agent had *before* this task.
    ```
    {old_memory}
    ```
4.  **Critique and Decide:**
    - Was the agent's strategy effective?
    - Is it a *new, generalizable* strategy that should be added?
    - Does it *refine* or *correct* a bad strategy in the old cheatsheet?
5.  **Produce the New Cheatsheet (M_i+1):** Rewrite and return the *entire* cheatsheet, incorporating any new, high-value, generalizable strategies. 
    - **Do not just append.** Prune bad or redundant rules. 
    - Keep it concise.
    - If no changes are needed, just return the old cheatsheet.
Return ONLY the text for the New, Updated Cheatsheet (M_i+1).
"""
