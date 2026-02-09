
import os
import re

files_to_process = [
    r"s:\Projects\object-detection-project\templates\live_upload.html",
    r"s:\Projects\object-detection-project\templates\saved_tests.html",
    r"s:\Projects\object-detection-project\templates\about.html",
    r"s:\Projects\object-detection-project\templates\help.html",
    r"s:\Projects\object-detection-project\templates\signup.html",
    r"s:\Projects\object-detection-project\templates\login.html"
]

def revert_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}: Not found")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove 'dark:' classes
    # Pattern: space or quote, then dark:, then chars until space or quote
    # Be careful not to remove the quote itself.
    # Regex: (?<=[\s"'])dark:[a-zA-Z0-9\-\/\[\]#,\(\)%]+
    
    # A safer way using re.sub with a callback or carefully crafted regex
    # We want to remove ` dark:something` (leading space) or `dark:something ` (trailing)
    # Most Tailwind classes are separated by spaces.
    
    content = re.sub(r'\s+dark:[a-zA-Z0-9\-\/\[\]#_,\(\)%]+', '', content)
    content = re.sub(r'dark:[a-zA-Z0-9\-\/\[\]#_,\(\)%]+\s+', '', content)
    content = re.sub(r'dark:[a-zA-Z0-9\-\/\[\]#_,\(\)%]+', '', content) # Remaining ones

    # 2. Remove Theme Toggle Button HTML Block
    # Look for the comment I added: <!-- Theme Toggle Button -->
    # and the button tag.
    
    if '<!-- Theme Toggle Button -->' in content:
        # Regex to match the comment and the button following it.
        # <button id="themeToggleBtn" ... </button>
        pattern = r'<!-- Theme Toggle Button -->\s*<button id="themeToggleBtn".*?</button>'
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    elif 'id="themeToggleBtn"' in content:
        # Fallback if comment is missing
        pattern = r'<button id="themeToggleBtn".*?</button>'
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # 3. Remove JS Logic
    # Pattern: // Theme Toggle Logic ... (until next distinct block or end of script)
    # I'll look for the specific lines I added.
    
    js_marker = "// Theme Toggle Logic"
    if js_marker in content:
        # Try to find the block. It usually ends before `// Auth check` or closing script tag styling.
        # In saved_tests/about/help, it was pasted together.
        
        # We'll use a specific logic: Find start index, find end index (e.g. `const signupForm` or `function checkAuth` or `</script>`)
        
        start_idx = content.find(js_marker)
        
        # Candidates for end of block
        end_markers = [
            "// Auth check", 
            "const signupForm =", 
            "const loginBtn =",
            "function showToast",
            "</script>"
        ]
        
        end_idx = -1
        nearest_marker = None
        
        # content[start_idx:]
        
        # We assume the logic is contained.
        # Actually, let's just remove specific lines if we can match them.
        pass

    # Alternative JS removal: specific replacement of the block
    js_block = r"""
    // Theme Toggle Logic
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeSun = document.getElementById('themeSun');
    const themeMoon = document.getElementById('themeMoon');

    // Check for saved user preference, if any, on load of the website
    const userTheme = localStorage.getItem('theme');
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Initial Theme Check
    if (userTheme === 'dark' || (!userTheme && systemTheme)) {
      document.documentElement.classList.add('dark');
      themeSun.classList.remove('hidden');
      themeMoon.classList.add('hidden');
    } else {
      document.documentElement.classList.remove('dark');
      themeSun.classList.add('hidden');
      themeMoon.classList.remove('hidden');
    }

    // Toggle Theme
    themeToggleBtn.addEventListener('click', () => {
      if (document.documentElement.classList.contains('dark')) {
        document.documentElement.classList.remove('dark');
        localStorage.setItem('theme', 'light');
        themeSun.classList.add('hidden');
        themeMoon.classList.remove('hidden');
      } else {
        document.documentElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
        themeSun.classList.remove('hidden');
        themeMoon.classList.add('hidden');
      }
    });
    """
    # Normalize whitespace for matching? 
    # Or just use regex loosely matching the variable names.
    
    js_pattern = r'// Theme Toggle Logic.*?themeToggleBtn\.addEventListener.*?\}\);\n'
    content = re.sub(js_pattern, '', content, flags=re.DOTALL)

    # Also remove the `const userTheme` check at the start of body in signup/login if present
    # <script>
    # // Theme Logic
    # const userTheme = ...
    # ...
    # }
    
    # Pattern used in signup.html:
    # // Theme Logic
    # const userTheme = localStorage.getItem('theme');
    # ...
    # document.documentElement.classList.remove('dark');
    # }
    
    js_pattern_2 = r'// Theme Logic\s+const userTheme =.*?document\.documentElement\.classList\.remove\(\'dark\'\);\s+\}'
    content = re.sub(js_pattern_2, '', content, flags=re.DOTALL)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Reverted {file_path}")

if __name__ == "__main__":
    for f in files_to_process:
        revert_file(f)
