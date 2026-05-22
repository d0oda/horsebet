import re

with open("/Users/ryfei.wang/.gemini/antigravity/brain/88f30814-4711-4d43-af75-678262868090/stitch.html", "r") as f:
    html = f.read()

# Extract the body contents
body_match = re.search(r'<body[^>]*>(.*?)<script>', html, re.DOTALL)
if body_match:
    body_content = body_match.group(1)
else:
    print("Could not find body")
    exit(1)

# Convert class= to className=
jsx = body_content.replace('class=', 'className=')
# Close img tags
jsx = re.sub(r'(<img[^>]+)(?<!/)>', r'\1 />', jsx)
# Close input tags
jsx = re.sub(r'(<input[^>]+)(?<!/)>', r'\1 />', jsx)
# Replace style="width: 98%" with style={{ width: '98%' }}
jsx = re.sub(r'style="width:\s*([^"]+)"', r"style={{ width: '\1' }}", jsx)

# Wrap in a React component
react_code = f"""\"\"\"
"use client";

import {{ useState, useEffect }} from "react";

export default function RacedayDashboard() {{
  return (
    <>
      {jsx}
    </>
  );
}}
\"\"\"
"""

with open("/Users/ryfei.wang/Documents/horsebet/frontend/app/page.tsx", "w") as f:
    f.write(react_code.replace('"""\n', '').replace('\n"""', ''))

print("Converted successfully.")
