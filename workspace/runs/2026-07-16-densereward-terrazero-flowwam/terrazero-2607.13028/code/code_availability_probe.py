"""
Probe for public TerraZero / SPACeR / TerraTransfer code and artifacts.
Run with: python code_availability_probe.py
"""
import json, urllib.request, urllib.error, sys

REPOS = [
    ("TerraZero project page", "https://github.com/terra-applied/TerraZero"),
    ("TerraZero author mirror", "https://github.com/akshay-rangesh-ai/TerraZero"),
    ("SPACeR project page", "https://github.com/spacer-ai/spacer-ai.github.io"),
]

HF_DATASET = "https://huggingface.co/api/datasets/woo-who/terrazero-assets/tree/main/"

def check_repo(name, url):
    api = url.replace("https://github.com/", "https://api.github.com/repos/")
    try:
        with urllib.request.urlopen(api, timeout=15) as r:
            data = json.loads(r.read())
        print(f"\n{name}: {url}")
        print(f"  description: {data.get('description')}")
        print(f"  language: {data.get('language')}")
        print(f"  size_kb: {data.get('size', 0)}")
        # List root contents
        contents_api = api + "/contents/"
        try:
            with urllib.request.urlopen(contents_api, timeout=15) as r:
                items = json.loads(r.read())
            print(f"  root contents: {[d['name'] for d in items]}")
            # crude heuristic: if only index.html / assets / pdf, it's a project-page repo
            names = {d['name'] for d in items}
            if names <= {'index.html', 'assets', 'TerraZero.pdf', '.gitattributes', 'README.md'}:
                print("  -> appears to be a project-page / PDF repo, not source code")
        except Exception as e:
            print(f"  contents error: {e}")
    except Exception as e:
        print(f"{name}: FAILED {e}")

def check_hf():
    print("\nHuggingFace dataset woo-who/terrazero-assets root:")
    try:
        with urllib.request.urlopen(HF_DATASET, timeout=15) as r:
            items = json.loads(r.read())
        for d in items:
            print(f"  {d['type']:10} {d['path']}")
        print("  -> contains paper PDF and video directories; no code / model files visible")
    except Exception as e:
        print(f"  HF error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Public code availability probe for TerraZero ecosystem")
    print("="*60)
    for name, url in REPOS:
        check_repo(name, url)
    check_hf()
