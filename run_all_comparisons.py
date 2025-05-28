import os
import subprocess
import json
import argparse
from pathlib import Path
import time

# --- Configuration ---
BASE_EXPERIMENT_DIR = "/raid/lawrence/substitution_rate_experiments"
TOP_P_VALUES = [0.9, 0.95, 0.98, 0.99, 0.995]
NUM_GENERATIONS_PER_CASE = 10 # Number of times to run for each prompt/top_p combination
MAX_PARALLEL_GPUS = 8

PROMPTS = {
    "book_reports": [
        "Write an extensive, winding summary and analysis of The Brothers Karamazov. It should be at least 2000 words long.",
        "Provide a detailed critique of '1984' by George Orwell, focusing on its themes of totalitarianism and surveillance. Aim for 1500-2000 words.",
        "Explore the concept of the American Dream in F. Scott Fitzgerald's 'The Great Gatsby'. Discuss its attainability and critique. Word count: 2000.",
        "Analyze the portrayal of heroism and morality in Homer's 'The Odyssey'. Discuss Odysseus's character flaws and strengths. Minimum 1800 words.",
        "Compare and contrast the societal critiques presented in Aldous Huxley's 'Brave New World' and Yevgeny Zamyatin's 'We'. At least 2000 words.",
        # "Discuss the themes of justice, revenge, and redemption in Alexandre Dumas's 'The Count of Monte Cristo'. Aim for 2000 words.",
        # "Examine the role of women and societal expectations in Jane Austen's 'Pride and Prejudice'. Provide a comprehensive analysis. 1500-2000 words.",
        # "Write a detailed literary analysis of Gabriel Garcia Marquez's 'One Hundred Years of Solitude', focusing on magical realism and family sagas. Minimum 2000 words.",
        # "Explore the themes of guilt, punishment, and redemption in Fyodor Dostoevsky's 'Crime and Punishment'. Word count: 2000.",
        # "Analyze the philosophical and existential themes in Albert Camus's 'The Stranger'. Discuss Meursault's detachment and the absurd. At least 1800 words."
    ],
    "story_generation": [
        "Write a complex political thriller that takes place in the year 3000, on a newly colonized Mars. There should be twists and turns and surprises. It should be at least 2000 words long.",
        "Craft a fantasy epic about a reluctant hero who discovers they are the last of an ancient line of dragon riders, tasked with preventing a continental war. Minimum 2500 words.",
        "Develop a science fiction narrative about a deep-space exploration crew that encounters a sentient, crystalline life form that communicates through complex light patterns, leading to a profound discovery about the universe. Aim for 2000 words.",
        "Write a historical fiction piece set during the Renaissance, focusing on an apprentice artist who gets entangled in a dangerous conspiracy involving a powerful patron and a hidden masterpiece. 1800-2200 words.",
        "Create a mystery story where a detective with a unique ability (e.g., to see recent memories of objects they touch) must solve a seemingly impossible murder in a technologically advanced, isolated research facility. At least 2000 words.",
        # "Compose a heartwarming tale about an old toymaker who crafts magical toys that come to life, and their adventure to save their town from a creeping apathy. Word count: 1500-2000.",
        # "Write a dystopian story where emotions are suppressed by technology, and a small group of rebels rediscovers art and music as a means of awakening society. Minimum 2000 words.",
        # "Develop a steampunk adventure involving a brilliant inventor and a daring airship pilot as they race against a shadowy organization to find a legendary power source. 2000-2500 words.",
        # "Craft a supernatural horror story about a group of paranormal investigators who spend a night in a notoriously haunted lighthouse, only to find the legends are terrifyingly real. Aim for 1800 words.",
        # "Write a comedic sci-fi story about a hapless intergalactic delivery pilot who accidentally gets involved in a mission to save the galaxy, armed only with a broken translator and a perpetually optimistic robot companion. At least 2000 words."
    ],
    "coding_prompts": [
        "Generate a Python script that implements a complete command-line based inventory management system. It should support adding items, removing items, updating quantities, searching for items, and generating reports (e.g., low stock, total inventory value). Use CSV files for data storage. Include comprehensive error handling and user-friendly interactions. The script should be well-documented with comments and a README file explaining its usage. Aim for a substantial codebase.",
        "Write a set of interacting Python classes to simulate a simplified version of a social media platform. Include classes for User, Post, Comment, and Feed. Users should be able to create posts, comment on posts, follow other users, and view a feed of posts from users they follow. Implement methods for these actions. Focus on clear class structure and interactions. Generate example usage. Minimum 300 lines of code.",
        "Develop a Python application that acts as a personal finance tracker. It should allow users to input income and expenses, categorize transactions, view spending summaries by category, and set budgets. Data should be stored persistently (e.g., using SQLite or JSON files). The application should have a clear command-line interface or a simple text-based UI. Generate a detailed plan and then the code.",
        "Create a Python script that scrapes data from a mock e-commerce website (you can define the HTML structure for 2-3 product pages and a category page). The script should extract product names, prices, descriptions, and image URLs. Store the scraped data in a structured format like JSON or a CSV file. Handle potential issues like missing data or changes in page structure gracefully. The script should be modular.",
        "Implement a Python program that simulates a simple version of the RSA encryption and decryption algorithm. Include functions for key generation (public and private keys), encryption of a message, and decryption of the ciphertext. Explain each step with comments. Provide an example of encrypting and decrypting a message. Focus on the mathematical operations involved. This should be a fairly long and detailed script.",
        # "Write a Python script that analyzes a large text file (e.g., a public domain book) to find the N most frequent words, their frequencies, and also the M most frequent K-grams (sequences of K words). The script should allow the user to specify N, M, and K. Exclude common stop words. Output the results in a readable format. Aim for efficiency with large files.",
        # "Develop a Python program that implements a basic pathfinding algorithm (e.g., A* or Dijkstra's) on a 2D grid. The grid can have obstacles. The program should take a start point, an end point, and the grid configuration as input, and output the shortest path or indicate if no path exists. Visualize the grid and path in the terminal if possible. Make the code object-oriented.",
        # "Generate Python code for a Flask web application that provides a simple API for a to-do list. The API should support creating, reading, updating, and deleting tasks (CRUD operations). Tasks should be stored in memory or a simple file for persistence. Include API documentation (e.g., using comments or a separate Markdown file). Write example client-side requests using `curl` or Python's `requests` library.",
        # "Write a Python script that simulates a multi-threaded download manager. The manager should be able to download multiple mock 'files' (represented by delaying functions) concurrently. It should display the progress of each download and a summary when all downloads are complete. Focus on threading, locking (if necessary for shared resources), and progress reporting. This will require careful design for concurrency.",
        # "Create a Python program that implements a Huffman coding algorithm for text compression. The program should take a string as input, build the Huffman tree, generate the codes for each character, encode the input string into its binary Huffman representation, and then be able to decode it back to the original string. Display the Huffman codes and the compression ratio achieved. This is a classic algorithm with several components to implement."
    ]
}

def check_experiment_completed(output_dir):
    """
    Check if an experiment has been completed by looking for watermark experiment directories.
    Returns True if at least one watermark experiment directory exists in the output_dir.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return False
    
    # Look for directories that match the watermark experiment pattern
    watermark_dirs = list(output_path.glob("watermark_*"))
    return len(watermark_dirs) > 0

def count_existing_experiments():
    """
    Count how many experiments have already been completed.
    """
    completed_count = 0
    total_count = 0
    
    base_dir = Path(BASE_EXPERIMENT_DIR)
    
    for category, prompt_list in PROMPTS.items():
        for i, _ in enumerate(prompt_list[:5]):
            prompt_dir_name = f"prompt_{i:02d}"
            for top_p in TOP_P_VALUES:
                for run_num in range(1, NUM_GENERATIONS_PER_CASE + 1):
                    total_count += 1
                    
                    top_p_str = str(top_p).replace('.', '_')
                    output_dir = base_dir / category / prompt_dir_name / f"top_p_{top_p_str}"
                    
                    if check_experiment_completed(output_dir):
                        completed_count += 1
    
    return completed_count, total_count

# --- Main Script ---
def run_experiments(resume=False):
    base_dir = Path(BASE_EXPERIMENT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base experiment directory: {base_dir.resolve()}")

    # Count existing experiments if resuming
    if resume:
        completed_count, total_count = count_existing_experiments()
        print(f"RESUME MODE: Found {completed_count} completed experiments out of {total_count} total.")
        if completed_count == total_count:
            print("All experiments already completed!")
            return
        print(f"Will skip {completed_count} completed experiments and run {total_count - completed_count} remaining.")
    
    active_processes = []
    gpu_ids = list(range(MAX_PARALLEL_GPUS))
    available_gpus = gpu_ids.copy()
    
    tasks = []
    skipped_count = 0
    
    for category, prompt_list in PROMPTS.items():
        # Take only the first 5 prompts per category
        for i, prompt_text in enumerate(prompt_list[:5]): 
            prompt_dir_name = f"prompt_{i:02d}"
            for top_p in TOP_P_VALUES:
                for run_num in range(1, NUM_GENERATIONS_PER_CASE + 1):
                    # Check if this experiment already exists when resuming
                    if resume:
                        top_p_str = str(top_p).replace('.', '_')
                        output_dir = base_dir / category / prompt_dir_name / f"top_p_{top_p_str}"
                        
                        if check_experiment_completed(output_dir):
                            skipped_count += 1
                            print(f"SKIPPING completed: Cat='{category}', Prmpt='{prompt_dir_name}', TopP={top_p}, Run={run_num}")
                            continue
                    
                    tasks.append({
                        "category": category,
                        "prompt_text": prompt_text,
                        "prompt_dir_name": prompt_dir_name,
                        "top_p": top_p,
                        "run_num": run_num
                    })
    
    if resume and skipped_count > 0:
        print(f"Skipped {skipped_count} already completed experiments.")
    
    task_idx = 0
    total_tasks = len(tasks)
    print(f"Total tasks to run: {total_tasks}")
    
    if total_tasks == 0:
        print("No tasks to run!")
        return

    while task_idx < total_tasks or active_processes:
        # Launch new processes if there are available GPUs and pending tasks
        while available_gpus and task_idx < total_tasks:
            gpu_id = available_gpus.pop(0)
            current_task = tasks[task_idx]
            task_idx += 1

            category = current_task["category"]
            prompt_text = current_task["prompt_text"]
            prompt_dir_name = current_task["prompt_dir_name"]
            top_p = current_task["top_p"]
            run_num = current_task["run_num"]

            category_dir = base_dir / category
            category_dir.mkdir(exist_ok=True)
            # Save prompt mapping once per category if it doesn't exist or needs update
            # For simplicity, always write it; it's small.
            prompt_mapping = {f"prompt_{idx:02d}": p_text for idx, p_text in enumerate(PROMPTS[category][:5])}
            mapping_file_path = category_dir / "prompt_mapping.json"
            with open(mapping_file_path, 'w') as f:
                json.dump(prompt_mapping, f, indent=4)

            prompt_category_base_dir = category_dir / prompt_dir_name
            prompt_category_base_dir.mkdir(exist_ok=True)

            top_p_str = str(top_p).replace('.', '_')
            output_dir_for_comparison_script = prompt_category_base_dir / f"top_p_{top_p_str}"
            output_dir_for_comparison_script.mkdir(parents=True, exist_ok=True)
            
            log_file_path = output_dir_for_comparison_script / f"run_{run_num:02d}_gpu_{gpu_id}.log"

            print(f"Launching task {task_idx}/{total_tasks}: Cat='{category}', Prmpt='{prompt_dir_name}', TopP={top_p}, Run={run_num} on GPU {gpu_id}")
            print(f"  Prompt: '{prompt_text[:50]}...'")
            print(f"  Output base for run: {output_dir_for_comparison_script.resolve()}")
            print(f"  Log file: {log_file_path}")

            command = [
                "./run_comparison.sh",
                str(top_p),
                prompt_text,
                str(output_dir_for_comparison_script.resolve())
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            with open(log_file_path, 'w') as log_file:
                process = subprocess.Popen(command, env=env, stdout=log_file, stderr=log_file)
            active_processes.append((process, gpu_id, current_task, log_file_path))

        # Check for completed processes
        for i in range(len(active_processes) - 1, -1, -1):
            process, gpu_id, task_info, log_path = active_processes[i]
            if process.poll() is not None: # Process has finished
                active_processes.pop(i)
                available_gpus.append(gpu_id)
                print(f"Task COMPLETED (GPU {gpu_id}): Cat='{task_info['category']}', Prmpt='{task_info['prompt_dir_name']}', TopP={task_info['top_p']}, Run={task_info['run_num']}. Log: {log_path}")
                if process.returncode != 0:
                    print(f"  WARNING: Task exited with code {process.returncode}. Check log for errors: {log_path}")
        
        if active_processes or task_idx < total_tasks:
            time.sleep(5) # Wait a bit before checking again

    print("\nAll experiments finished.")
    print(f"Results are in: {base_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run watermarking experiments across different prompts and top_p values")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from existing experiments, skipping already completed ones")
    
    args = parser.parse_args()
    
    run_experiments(resume=args.resume) 