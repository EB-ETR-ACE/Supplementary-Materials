# --- agent_v1.py ---
# Refactored 2-node agent (Generator -> Curator)
# Replaces functionality of agent-v0 and agent-v1

import logging
import argparse
import sys
from typing import TypedDict, Annotated, Optional
from pathlib import Path

# Core LangGraph imports
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Pydantic import
from pydantic import BaseModel, Field

# The Ollama import
from langchain_ollama import ChatOllama

# Local Config
try:
    from inference_config import (
        DEFAULT_OLLAMA_URL, 
        GENERATOR_PROMPT_TEMPLATE as DEFAULT_GEN_PROMPT,
        CURATOR_V1_PROMPT_TEMPLATE as DEFAULT_CUR_PROMPT
    )
except ImportError:
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_GEN_PROMPT = "Error: Config not found."
    DEFAULT_CUR_PROMPT = "Error: Config not found."

# --- Logging Setup ---
def setup_logging(log_file: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

# --- State & Schemas ---
class GermanSimplificationState(TypedDict):
    query: str
    target_level: str
    memory: str
    current_solution: str
    current_strategy: str
    # No reflection_insights in V1
    messages: Annotated[list, add_messages]

class GeneratorOutput(BaseModel):
    strategy: str = Field(description="The step-by-step strategy used.")
    solution: str = Field(description="The final simplified German text.")

# --- Nodes ---

def get_llm(model_name: str, base_url: str):
    return ChatOllama(model=model_name, base_url=base_url)

def generate_solution(state: GermanSimplificationState, config: dict):
    logging.info("--- 🧠 Calling Generator (V1) ---")
    query = state['query']
    target_level = state['target_level']
    memory = state['memory']
    
    prompt_template = config.get('gen_prompt', DEFAULT_GEN_PROMPT)
    llm = config['llm']
    structured_llm = llm.with_structured_output(GeneratorOutput)

    prompt = prompt_template.format(
        memory=memory, query=query, target_level=target_level
    )
    
    try:
        response = structured_llm.invoke(prompt)
        logging.info(f"Generator Strategy: {response.strategy}")
        logging.info(f"Generator Solution: {response.solution}")
        return {"current_solution": response.solution, "current_strategy": response.strategy}
    except Exception as e:
        logging.error(f"❌ ERROR in Generator: {e}", exc_info=True)
        return {"current_solution": f"GENERATOR_ERROR: {e}", "current_strategy": "Error"}

def curate_memory(state: GermanSimplificationState, config: dict):
    if "GENERATOR_ERROR" in state.get("current_solution", ""):
        return {"memory": state['memory']}

    logging.info("--- ✍️ Calling Curator (V1 - Implicit Reflection) ---")
    
    old_memory = state['memory']
    query = state['query']
    solution = state['current_solution']
    strategy = state['current_strategy']
    
    prompt_template = config.get('cur_prompt', DEFAULT_CUR_PROMPT)
    llm = config['llm']

    prompt = prompt_template.format(
        old_memory=old_memory, query=query, strategy=strategy, solution=solution
    )
    
    try:
        response = llm.invoke(prompt)
        new_memory = response.content
        if not new_memory or len(new_memory) < 10:
             return {"memory": old_memory}
        logging.info(f"Updated Memory: {new_memory}")
        return {"memory": new_memory}
    except Exception as e:
        logging.error(f"❌ ERROR in Curator: {e}", exc_info=True)
        return {"memory": old_memory}

# --- Main Graph Builder ---

def build_graph(model_name: str, ollama_url: str, prompts: dict):
    llm = get_llm(model_name, ollama_url)
    config = {'llm': llm, **prompts}

    builder = StateGraph(GermanSimplificationState)
    
    builder.add_node("generator", lambda state: generate_solution(state, config))
    builder.add_node("curator", lambda state: curate_memory(state, config))

    builder.set_entry_point("generator")
    builder.add_edge("generator", "curator")
    builder.add_edge("curator", END)
    
    return builder

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="ACE Agent V1 (2-node)")
    parser.add_argument("-i", "--input", required=True, help="Input text file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-l", "--level", required=True, help="Target level")
    parser.add_argument("--model-name", required=True, help="Ollama model name")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama URL")
    parser.add_argument("--db-path", default=":memory:", help="SQLite DB path")
    parser.add_argument("--seed-memory", help="Path to seed memory file")
    parser.add_argument("--log-file", help="Log file path")
    
    parser.add_argument("--generator-prompt-file", help="Path to generator prompt file")
    parser.add_argument("--curator-prompt-file", help="Path to curator prompt file")

    args = parser.parse_args()

    setup_logging(args.log_file)
    
    prompts = {
        'gen_prompt': Path(args.generator_prompt_file).read_text(encoding='utf-8') if args.generator_prompt_file else DEFAULT_GEN_PROMPT,
        'cur_prompt': Path(args.curator_prompt_file).read_text(encoding='utf-8') if args.curator_prompt_file else DEFAULT_CUR_PROMPT,
    }

    builder = build_graph(args.model_name, args.ollama_url, prompts)
    
    conn_string = args.db_path if args.db_path == ":memory:" else str(Path(args.db_path).resolve())
    
    try:
        with SqliteSaver.from_conn_string(conn_string) as checkpointer:
            app = builder.compile(checkpointer=checkpointer)
            
            try:
                query_text = Path(args.input).read_text(encoding='utf-8')
            except Exception as e:
                logging.error(f"Could not read input: {e}")
                sys.exit(1)

            thread_id = "agent_v1_run"
            config = {"configurable": {"thread_id": thread_id}}
            
            is_first_run = True
            try:
                current = checkpointer.get(config)
                if current and current.get("channel_values", {}).get("memory"):
                    is_first_run = False
            except: pass

            inputs = {"query": query_text, "target_level": args.level}
            
            if is_first_run:
                initial_mem = "Use simple words. Keep sentences short."
                if args.seed_memory and Path(args.seed_memory).is_file():
                    initial_mem = Path(args.seed_memory).read_text(encoding='utf-8')
                inputs["memory"] = initial_mem

            result = app.invoke(inputs, config=config)
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.get('current_solution', 'Error: No solution'), encoding='utf-8')

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
