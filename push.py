#!/usr/bin/env python3
"""Push windows-distributed-inference to GitHub"""
import os, subprocess, json, urllib.request, sys

# Read token from .env
env = os.path.expandvars(r'%LOCALAPPDATA%\hermes\.env')
token = None
with open(env) as f:
    for line in f:
        if line.startswith('GITHUB_TOKEN=***            token = line.split('=', 1)[1].strip()
            break

if not token:
    print("Set GITHUB_TOKEN in .env first")
    sys.exit(1)

repo_dir = os.path.expanduser('~/projects/distributed-inference')
os.chdir(repo_dir)

# Create repo
data = json.dumps({
    "name": "windows-distributed-inference",
    "description": "Windows双机异构GPU分布式推理：AMD RX6600+NVIDIA RTX4070跑14B 20K",
    "private": False,
    "auto_init": False
}).encode()

req = urllib.request.Request(
    'https://api.github.com/user/repos',
    data=data,
    headers={
        'Authorization': f'token {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'hermes'
    },
    method='POST'
)

try:
    resp = urllib.request.urlopen(req, timeout=15)
    result = json.loads(resp.read())
    print(f"Repo: {result['html_url']}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    if 'already exists' in body:
        print("Repo exists, pushing...")
    else:
        print(f"Error: {body}")
        sys.exit(1)

# Git operations
subprocess.run(['git', 'add', '-A'], check=True, cwd=repo_dir)
subprocess.run(['git', 'commit', '-m', 'docs: Windows异构GPU分布式推理完整实战记录'], check=True, cwd=repo_dir)

remote = f"https://{token}@github.com/yvi/windows-distributed-inference.git"
subprocess.run(['git', 'remote', 'add', 'origin', remote], cwd=repo_dir)
subprocess.run(['git', 'push', '-u', 'origin', 'main', '-f'], check=True, cwd=repo_dir)
print("Pushed!")
