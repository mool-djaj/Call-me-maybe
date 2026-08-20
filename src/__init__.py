import argparse
import json
import sys
from pathlib import Path
from typing import List, Any

# We import the Pydantic models we just created
from src.models import FunctionDef, PromptInput, FunctionCallOutput

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
    """Safely loads a JSON file using a context manager."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def main() -> None:
    args = parse_arguments()

    try:
        print(f"Loading function definitions from {args.functions_definition}...")
        raw_funcs = load_json_file(args.functions_definition)
        
        # Pydantic validates every function definition in the list
        functions: List[FunctionDef] = [FunctionDef(**func) for func in raw_funcs]

        print(f"Loading prompts from {args.input}...")
        raw_inputs = load_json_file(args.input)
        
        # Pydantic validates every prompt
        prompts: List[PromptInput] = [PromptInput(**p) for p in raw_inputs]

        print("Initializing the Generation Pipeline...")
        results: List[FunctionCallOutput] = []

        # --- THIS IS WHERE OUR GENERATOR WILL GO ---
        for prompt_data in prompts:
            print(f"\nProcessing prompt: '{prompt_data.prompt}'")
            
            # TODO: Pass the prompt and the validated functions to the LLM engine here.
            # For now, we create a dummy Pydantic result to verify the architecture works.
            dummy_result = FunctionCallOutput(
                prompt=prompt_data.prompt,
                name=functions[0].name if functions else "unknown",
                parameters={}
            )
            results.append(dummy_result)
        # -------------------------------------------

        # Save the validated output gracefully
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump([res.model_dump() for res in results], file, indent=4)
            
        print(f"\nSuccess! Output written to {args.output}")

    except Exception as e:
        # Graceful error handling as mandated by the subject
        print(f"\n[Error] The program encountered an issue: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()