import requests
from requests.utils import quote
from tqdm import tqdm
import json
import os
import re

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# Provide a GitHub personal access token via environment to handle authenticated search.
HEADERS = {"Authorization": f"Token {GITHUB_TOKEN}"} if GITHUB_TOKEN and GITHUB_TOKEN != "<YOUR_GITHUB_TOKEN_HERE>" else {}

# Allowed open-source licenses (can expand as needed)
ALLOWED_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "cc-by-4.0", "unlicense"}

# (Optional) Known task descriptions to exclude (HumanEval + MBPP to avoid duplicates)
EXCLUDE_DESCRIPTIONS = {
    "Return True if list elements are monotonically increasing or decreasing", 
    "Write a function to check if a string is a palindrome", 
    # ... (Load or list all HumanEval/MBPP docstring prompts here for filtering) ...
}

def search_github_code(query, max_results=1000):
    """Search GitHub code for a given query. Returns a list of file info dicts."""
    results = []
    url = "https://api.github.com/search/code"
    # Paginate results 100 per page
    for page in range(1, (max_results // 100) + 2):
        params = {"q": query, "per_page": 100, "page": page}
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code != 200:
            break  # Stop on any error or rate limit
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        results.extend(items)
        if len(items) < 100:
            break  # no more pages
    return results

def get_repo_license(repo_full_name):
    """Return the license key of the given repository (e.g., 'mit'). If none or not permissive, return None."""
    repo_url = f"https://api.github.com/repos/{repo_full_name}"
    resp = requests.get(repo_url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    info = resp.json()
    license_info = info.get("license")
    if license_info:
        key = license_info.get("spdx_id", "").lower()  # use SPDX identifier for license
        return key
    return None

def extract_functions_from_file(file_content):
    """Extract (function_signature, docstring) pairs from a Python file content string."""
    funcs = []
    lines = file_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("def ") and line.endswith(":"):
            signature = line  # capture the full function signature line
            # Only consider top-level functions (avoid indent or methods inside classes)
            if signature.startswith("def "):
                # Look for triple-quoted docstring right after function definition
                j = i + 1
                # Skip possible decorators or blank lines between signature and docstring
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('"""'):
                    doc_lines = []
                    # Collect lines until closing triple quote
                    doc_lines.append(lines[j].strip().strip('"""'))  # first line without quotes
                    j += 1
                    while j < len(lines) and '"""' not in lines[j]:
                        doc_lines.append(lines[j].strip())
                        j += 1
                    if j < len(lines) and '"""' in lines[j]:
                        # add last part of docstring on the same line as closing quotes
                        doc_end_part = lines[j].split('"""')[0]
                        doc_lines.append(doc_end_part.strip())
                    docstring = " ".join(doc_lines).strip()
                    funcs.append((signature, docstring))
                    i = j  # jump to end of docstring
        i += 1
    return funcs

# Optional: function to vary wording (synonyms/templates) for synthetic tasks
TEMPLATES = [
    "Return {}",
    "Given {}, return {}",
    "Write a function that {}",
    "Compute {}",
    "{}"  # sometimes the description is fine as is
]
def vary_docstring(desc):
    """Randomly apply a template to the given description."""
    import random
    tpl = random.choice(TEMPLATES)
    # If template expects two placeholders but we have one, handle accordingly:
    if "{}" in tpl and tpl.count("{}") == 1:
        return tpl.format(desc)
    elif tpl.count("{}") == 2:
        # Split the description into two parts if possible (naively)
        parts = desc.split(" to ", 1)
        if len(parts) == 2:
            return tpl.format(parts[0], parts[1])
        else:
            # If we cannot split, just fill first and reuse for second
            return tpl.format(desc, desc)
    else:
        return desc

def generate_synthetic_tasks(n=100):
    """Generate n synthetic tasks (function signature and docstring)."""
    synthetic_tasks = []
    # Example pool of simple task descriptions and function names
    simple_tasks = [
        ("def is_prime(n: int):", "check if n is a prime number and return True if yes, False otherwise"),
        ("def reverse_string(s: str):", "return the reverse of the string s"),
        ("def factorial(n: int):", "compute the factorial of n"),
        ("def sum_list(numbers: list):", "return the sum of all elements in the list numbers"),
        ("def is_palindrome(s: str):", "return True if string s reads the same forwards and backwards")
        # ... add more base tasks or even generate variations ...
    ]
    import random
    # Ensure we have at least n tasks by cycling or combining basic templates
    while len(synthetic_tasks) < n:
        sig, desc = random.choice(simple_tasks)
        # Capitalize docstring first word occasionally
        if random.random() < 0.3:
            desc = desc[0].upper() + desc[1:]
        # Possibly add a period at end of description
        if random.random() < 0.5 and not desc.endswith('.'):
            desc = desc + "."
        # Vary wording using templates
        varied_desc = vary_docstring(desc)
        # Construct a proper docstring sentence
        docstring = varied_desc[0].upper() + varied_desc[1:]
        if not docstring.endswith("."):
            docstring += "."
        # Only add if not a duplicate description
        if docstring not in EXCLUDE_DESCRIPTIONS:
            synthetic_tasks.append({"function": sig, "docstring": docstring})
            EXCLUDE_DESCRIPTIONS.add(docstring)  # avoid adding same again
    return synthetic_tasks[:n]

# Main script execution
if __name__ == "__main__":
    all_tasks = []

    # 1. Generate synthetic tasks (for diversity and easy tasks)
    SYNTHETIC_COUNT = 300  # e.g., generate 200 synthetic tasks
    print(f"Generating {SYNTHETIC_COUNT} synthetic tasks...")
    synthetic_tasks = generate_synthetic_tasks(SYNTHETIC_COUNT)
    all_tasks.extend(synthetic_tasks)

    # 2. Prepare search queries for GitHub code search
    queries = [
        "\"Return the\" in:file language:Python",
        "\"Write a function\" in:file language:Python",
        "\"Given a\" in:file language:Python",
        "\"Determine whether\" in:file language:Python",
        "\"Compute the\" in:file language:Python",
        "\"Check if\" in:file language:Python",
        "def+is_ in:file language:Python",
        "def+has_ in:file language:Python",
        "def+find_ in:file language:Python",
        "def+check_ in:file language:Python",
        "def+compute_ in:file language:Python",
        "def+convert_ in:file language:Python"
        # ... add more patterns or phrases to cover different types ...
    ]

    print(f"Searching GitHub code for tasks using {len(queries)} queries...")
    for query in tqdm(queries, desc="GitHub Queries"):
        try:
            results = search_github_code(query)
        except Exception as e:
            print(f"Error during GitHub search for query '{query}': {e}")
            continue
        # Process each search result
        for item in results:
            repo_full = item.get("repository", {}).get("full_name")
            path = item.get("path")
            if not repo_full or not path:
                continue
            # Filter by repo license
            license_key = get_repo_license(repo_full)
            if license_key is None or license_key not in ALLOWED_LICENSES:
                continue  # skip if no license info or not in allowed list
            # Fetch file content from GitHub
            raw_url = f"https://raw.githubusercontent.com/{repo_full}/master/{quote(path)}"
            resp = requests.get(raw_url, headers=HEADERS)
            if resp.status_code != 200:
                raw_url = f"https://raw.githubusercontent.com/{repo_full}/main/{quote(path)}"
                resp = requests.get(raw_url, headers=HEADERS)
            if resp.status_code != 200:
                continue  # skip if file not accessible
            file_text = resp.text
            # Extract function-docstring pairs
            functions = extract_functions_from_file(file_text)
            for sig, doc in functions:
                if not doc or len(doc) < 5:
                    continue
                # Simple filtering: avoid builtin or very trivial names
                if sig.startswith("def __") or sig.startswith("def _"):
                    continue  # skip dunder or private functions
                # Clean up docstring text (remove any trailing punctuation, excessive whitespace)
                doc_clean = doc.strip()
                # Avoid duplicates or excluded tasks
                if doc_clean in EXCLUDE_DESCRIPTIONS:
                    continue
                # Exclude if doc seems to reference LeetCode or similar (to avoid copyrighted descriptions)
                if "LeetCode" in doc_clean or "Project Euler" in doc_clean:
                    continue
                # Finally, add to tasks
                all_tasks.append({"function": sig, "docstring": doc_clean})
                EXCLUDE_DESCRIPTIONS.add(doc_clean)
        # End of results loop
    # End of query loop

    # 3. If using APPS or other dataset (pseudo-code, as an optional step)
    # Here we assume we have a function to load APPS problems if desired.
    # def load_apps_problems(max_count): ...
    # apps_tasks = load_apps_problems(5000)  # e.g., take 5000 from APPS
    # all_tasks.extend(apps_tasks)

    # 4. Finalize the dataset
    # Shuffle tasks for randomness
    import random
    random.shuffle(all_tasks)
    # Trim or pad to reach ~10000 tasks
    dataset = all_tasks[:15000]
    print(f"Collected {len(dataset)} tasks. Saving to JSON...")

    # 5. Save to JSON file
    with open("coding_15000_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
