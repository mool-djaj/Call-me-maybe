import argparse
import json
import sys
from pathlib import Path
from typing import List, Any

# Import the 1337-provided SDK
from llm_sdk import Small_LLM_Model

# Import our strict Pydantic structs and custom engine
from src.models import FunctionDef, PromptInput, FunctionCallOutput
from src.generator import generate_constrained_json

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments according to project requirements."""
    parser = argparse.ArgumentParser(description="Constrained Function Calling LLM Tool")
    parser.add_argument(
        "--functions_definition", 
        type=str, 
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing available functions."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing the test prompts."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/function_calling_results.json",
        help="Path to save the generated JSON function calls."
    )
    return parser.parse_args()

def load_json_file(filepath: str) -> Any:
    """Safely loads a JSON file using a context manager to prevent leaks."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def main() -> None:
    args = parse_arguments()

    try:
        # 1. Load and strictly validate the function schemas
        print(f"Loading function definitions from {args.functions_definition}...")
        raw_funcs = load_json_file(args.functions_definition)
        functions: List[FunctionDef] = [FunctionDef(**func) for func in raw_funcs]

        # 2. Load and validate the human prompts
        print(f"Loading prompts from {args.input}...")
        raw_inputs = load_json_file(args.input)
        prompts: List[PromptInput] = [PromptInput(**p) for p in raw_inputs]

        # 3. Boot up the AI Model
        print("Initializing the AI Model into memory...")
        model = Small_LLM_Model()
        results: List[FunctionCallOutput] = []

        # 4. The Main Execution Loop
        for prompt_data in prompts:
            print(f"\nProcessing prompt: '{prompt_data.prompt}'")
            
            # Run our math hacker engine to force the perfect JSON string
            raw_json_string = generate_constrained_json(model, prompt_data.prompt, functions)
            
            print(f"Raw engine output: {raw_json_string}")
            
            # Parse the strict string back into a Python dictionary
            parsed_data = json.loads(raw_json_string)
            
            # Pass it through the final Pydantic struct to guarantee it meets 1337 specs
            final_result = FunctionCallOutput(
                prompt=prompt_data.prompt,
                name=parsed_data["name"],
                parameters=parsed_data.get("parameters", {})
            )
            results.append(final_result)

        # 5. Save the final payload to disk
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as file:
            # We use .model_dump() to convert the strict structs back to standard dicts
            json.dump([res.model_dump() for res in results], file, indent=4)
            
        print(f"\nSuccess! All outputs strictly validated and written to {args.output}")

    except Exception as e:
        # The project explicitly forbids unexpected crashes. This catches everything.
        print(f"\n[Fatal Error] The program halted cleanly: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()